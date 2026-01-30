# Lightweight depth viewer for the DP3/VLA message format.
# Listens for the same ZMQ payloads that dp3_vla_inference.py consumes,
# prints depth stats, and shows a debug window on the dev machine.

import argparse
import io
import pickle
import time

import numpy as np
import zmq
from PIL import Image

def decompress_sample(sample: dict):
    """
    Decode compressed image/depth arrays in-place (same convention as dp3_vla_inference).
    """
    if sample.get("compressed", False):
        for key in ["image_array", "image_wrist_array"]:
            if key in sample:
                sample[key] = [np.array(Image.open(io.BytesIO(buf))) for buf in sample[key]]
        if "depth_array" in sample:
            sample["depth_array"] = [
                np.array(Image.open(io.BytesIO(buf))).view(np.float32).squeeze(axis=-1)
                for buf in sample["depth_array"]
            ]
        sample["compressed"] = False

def print_depth_stats(depth: np.ndarray, prefix: str = ""):
    """
    Print simple stats for a depth map.
    """
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = d[d > 0]
    valid_count = valid.size
    total = d.size
    if valid_count == 0:
        print(f"{prefix} valid_px=0 / {total} (0.0%)")
        return
    print(
        f"{prefix} valid_px={valid_count} / {total} ({valid_count/total*100:.2f}%) "
        f"min={valid.min():.4f} max={valid.max():.4f} mean={valid.mean():.4f}"
    )

def main():
    parser = argparse.ArgumentParser(description="Depth debug viewer for DP3/VLA messages")
    parser.add_argument("--port", type=int, required=True, help="ZMQ port to bind (same as dp3_vla_inference)")
    parser.add_argument("--print-image", dest="print_image", action="store_true",
                        help="Print a short notice when image frames are present")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{args.port}")
    print(f"[depth_view] listening on tcp://*:{args.port}")
    
    

    try:
        while True:
            client_id, empty, data = socket.recv_multipart()
            sample = pickle.loads(data)
            print("[depth_view] keys:", sorted(sample.keys()))
            decompress_sample(sample)
            print("[depth_view] keys:", sorted(sample.keys()))


            depth_seq = sample.get("depth_array", [])
            if len(depth_seq) == 0:
                print("[depth_view] received sample with no depth_array")
                depth = None
            else:
                depth = depth_seq[-1]
                print_depth_stats(depth, prefix=f"[depth_view] env={sample.get('env_id')}")

            if args.print_image:
                img_seq = sample.get("image_array") or sample.get("image_wrist_array")
                if img_seq:
                    arr = img_seq[-1]
                    print(f"[depth_view] image detected shape={arr.shape} dtype={arr.dtype}")

            # send ack so senders don't block waiting
            socket.send_multipart([
                client_id,
                b'',
                pickle.dumps({"info": "depth_view_ok", "env_id": sample.get("env_id")})
            ])

    except KeyboardInterrupt:
        print("[depth_view] stopping on Ctrl+C")
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()

