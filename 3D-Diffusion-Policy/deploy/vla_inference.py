# FIXME: import urchin before others, otherwise segfault, unknown reason
from urchin import URDF, Collision, Sphere, Geometry  # type: ignore  # must stay first

import os  # stdlib for env + paths
if 'DEBUG_PORT' in os.environ:  # optional remote debugging hook
    import debugpy  # debugger entry
    debugpy.listen(int(os.environ['DEBUG_PORT']))  # bind debugger port
    print('waiting for debugger to attach...')  # notify operator
    debugpy.wait_for_client()  # block until debugger connected

import argparse  # CLI parsing
arg_parser = argparse.ArgumentParser()  # build parser instance
arg_parser.add_argument("--port", type=str, required=True)  # ZMQ service port
arg_parser.add_argument("--batching-delay", type=float, default=80)  # ms before forcing batch
arg_parser.add_argument("--batch-size", type=int, default=1)  # max reqs per batch
arg_parser.add_argument("--dataset-statistics", type=str, required=True)  # reserved flag (unused here)
arg_parser.add_argument("--path", type=str, required=True)  # checkpoint path for VLA
arg_parser.add_argument("--openloop", action="store_true")  # open-loop toggle (unused)
arg_parser.add_argument("--top_k", type=int, default=None)  # optional top-k filter
arg_parser.add_argument("--compile", action="store_true")  # enable torch.compile

import PIL  # image decoding
import io  # bytes buffer
from typing import List  # type hints
import zmq  # networking
import pickle  # serialization
import time  # timing utilities
import numpy as np  # numerical ops
from tqdm import tqdm  # progress meter
from vla_network.model.vla.agent import VLAAgent  # VLA agent class
from vla_network.datasample.vla import VLASample  # VLA sample wrapper
import torch  # ML backend
torch.autograd.set_grad_enabled(False)  # force inference-only mode

from gx_utils.logger import log  # lightweight logger


def interpolate_delta_actions(delta_actions, n):
    """
    Upsample coarse delta actions by factor n to smooth execution.
    Each action: (dx, dy, dz, droll, dpitch, dyaw, gripper).
    """
    import transforms3d as t3d  # local import to avoid startup overhead
    ret = []  # collected fine-grained actions
    for delta_action in delta_actions:  # loop per coarse step
        xyzs = 1 / n * np.array([delta_action[:3]] * n)  # split translation evenly
        axangle_ax, axangle_angle = t3d.euler.euler2axangle(*delta_action[3:6])  # Euler→axis/angle
        eulers = [t3d.euler.axangle2euler(axangle_ax, axangle_angle / n)] * n  # split rotation
        grippers = np.array([[0.]] * (n - 1) + [[delta_action[-1]]])  # gripper only final step
        ret.extend(np.concatenate([xyzs, eulers, grippers], axis=-1))  # add expanded step
    return ret  # list of length m*n


def batch_process(vla_model: VLAAgent, batch: List[dict]):
    """
    Decode inputs, build VLASample batch, run model, postprocess actions.
    """
    for sample in batch:  # per incoming request
        if sample.get('compressed', False):  # decode if compressed
            for key in ['image_array', 'image_wrist_array']:  # RGB keys
                decompressed_image_array = []  # holder
                for compressed_image in sample[key]:  # each frame
                    decompressed_image_array.append(np.array(PIL.Image.open(io.BytesIO(compressed_image))))  # decode image
                sample[key] = decompressed_image_array  # replace with np arrays
            if 'depth_array' in sample:  # optional depth frames
                decompressed_depth_array = []  # holder
                for compressed_depth in sample['depth_array']:
                    decompressed_depth_array.append(
                        np.array(PIL.Image.open(io.BytesIO(compressed_depth))).view(np.float32).squeeze(axis=-1)
                    )  # float depth map
                sample['depth_array'] = decompressed_depth_array  # store decoded depth
            sample['compressed'] = False  # mark done

    assert vla_model.config.model.image_keys == ["left", "right"]  # assumes two views
    dt_steps = round(vla_model.config.model.dt / 0.1)  # upsample factor vs 10Hz ctrl
    frame_indices = list(range(-(vla_model.config.model.proprio_steps - 1) * dt_steps - 1, 0, dt_steps))  # history offsets
    input_batch = []  # VLASample list
    for sample in batch:  # construct per-request sample
        input_batch.append(VLASample(
            dataset_name="agent",  # synthetic dataset tag
            embodiment='epiclab_franka',  # robot type id
            frame=0,  # current frame index
            instruction=sample['text'],  # natural language command
            images=dict(
                left=[PIL.Image.fromarray(sample['image_wrist_array'][i]) for i in frame_indices[-vla_model.config.model.image_steps:]],  # wrist cam seq
                right=[PIL.Image.fromarray(sample['image_array'][i]) for i in frame_indices[-vla_model.config.model.image_steps:]],  # main cam seq
            ),
            proprio=np.array([sample['proprio_array'][i] for i in frame_indices]),  # proprio history
        ))
    results: List[VLASample] = vla_model(input_batch)  # forward pass
    ret = []  # aggregated responses
    for result, input_sample in zip(results, batch):  # align outputs to inputs
        action = result.action  # raw model deltas
        last_dim = action[:, -1]  # gripper logits
        last_dim = np.where(last_dim < -0.5, -1, np.where(last_dim > 0.5, 1, 0))  # discretize gripper
        action = np.concatenate([action[:, :-1], last_dim[:, None]], axis=-1)  # replace last column
        action = interpolate_delta_actions(action, dt_steps)  # temporal upsample
        debug = {}  # optional debug info
        if result.goal is not None:  # planned pose
            debug['pose'] = (result.goal[:3], result.goal[3:6])  # position, orientation
        if result.bboxs is not None:  # detections
            debug['bbox'] = [result.bboxs[k][-1] for k in vla_model.config.model.image_keys]  # last-frame bbox per view
        ret.append({
            'result': action,  # action sequence
            'env_id': input_sample['env_id'],  # route back to caller
            'debug': debug,  # extra info
        })
    return ret  # list of dicts ready for transport


def warmup(vla_model: VLAAgent):
    """
    Run a few dummy inferences to stabilize latency (CUDA graph / JIT warm).
    """
    SAMPLES = [  # minimal synthetic samples
        {
            'text': 'pick up elephant',  # dummy instruction
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])] * 4,
            'traj_metadata': None,
            'env_id': 1,
        },
        {
            'text': 'pick up toy large elephant',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])] * 4,
            'traj_metadata': None,
            'env_id': 2,
        },
        {
            'text': 'pick up toy car',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])] * 4,
            'traj_metadata': None,
            'env_id': 3,
        },
    ]
    NUM_TESTS = 5  # repeat count
    print('warming up...')  # notify
    for i in tqdm(range(NUM_TESTS)):  # initial warmup loop
        _ = batch_process(vla_model, [SAMPLES[i % len(SAMPLES)]])  # throw away result
    print('check the latency after warm up:')  # next stage
    for i in tqdm(range(NUM_TESTS)):  # measure stabilized latency
        _ = batch_process(vla_model, [SAMPLES[i % len(SAMPLES)]])  # throw away result


def main():
    args = arg_parser.parse_args()  # parse CLI
    vla_model = VLAAgent(args.path, compile=args.compile)  # load model (optionally compiled)

    warmup(vla_model)  # pre-run to stabilize performance

    context = zmq.Context()  # ZMQ context
    socket = context.socket(zmq.ROUTER)  # ROUTER allows addressing clients
    socket.bind(f"tcp://*:{args.port}")  # bind to requested port

    requests = []  # queued (client_id, payload)
    first_arrive_time = None  # timestamp of first queued item

    log.info('start serving')  # service start
    while True:  # server loop
        current_time = time.time() * 1000  # ms since epoch
        if (len(requests) >= args.batch_size or  # batch size reached
            (first_arrive_time is not None and  # or timeout expired
             current_time - first_arrive_time > args.batching_delay and
             len(requests) > 0)):
            data_num = min(args.batch_size, len(requests))  # number to process
            client_ids, data_batch = zip(*requests[:data_num])  # slice batch

            tbegin = time.time()  # measure processing
            log.info(f'start processing {len(requests)} requests')  # log queue depth
            results = batch_process(vla_model, data_batch)  # run inference
            tend = time.time()  # end time
            log.info(f'finished {len(requests)} requests in {tend - tbegin:.3f}s')  # duration

            for client_id, result in zip(client_ids, results):  # send replies
                socket.send_multipart([
                    client_id,  # routing id
                    b'',  # delimiter frame
                    pickle.dumps({
                        'info': 'success',  # status
                        'env_id': result['env_id'],  # echo env id
                        'result': result['result'],  # action sequence
                        'debug': result['debug'],  # extra info
                    })
                ])

            requests = requests[data_num:]  # drop processed
            if len(requests) == 0:  # reset timer
                first_arrive_time = None

        # try getting new sample
        try:
            client_id, empty, data = socket.recv_multipart(zmq.DONTWAIT)  # non-blocking recv
            if len(requests) == 0:
                first_arrive_time = time.time() * 1000  # mark first arrival

            data = pickle.loads(data)  # deserialize payload
            requests.append((client_id, data))  # enqueue
        except zmq.Again:  # no message
            pass  # loop back


if __name__ == "__main__":  # entrypoint guard
    main()  # start server
