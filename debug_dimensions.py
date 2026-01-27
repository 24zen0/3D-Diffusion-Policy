#!/usr/bin/env python3
"""
Debug script to check tensor dimensions for DP3 training
This script will help identify dimension mismatches in trajectory tensors
"""


import sys
import os
sys.path.append('/mnt/home/zengyitao/DP3baseline/3D-Diffusion-Policy/3D-Diffusion-Policy')

import hydra
import torch
from omegaconf import OmegaConf, ListConfig
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"__builtins__": {}}, {}))


from torch.utils.data import DataLoader
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

def debug_dimensions():
    # Load the configuration
    config_path = os.path.join('/mnt/home/zengyitao/DP3baseline/3D-Diffusion-Policy/3D-Diffusion-Policy', "diffusion_policy_3d/config")
    
    # Load the main config with the elephant6w task
    cfg = OmegaConf.load(os.path.join(config_path, "dp3.yaml"))
    task_cfg = OmegaConf.load(os.path.join(config_path, "task/elephant6w.yaml"))
    
    # Merge the configs
    cfg.task = task_cfg
    
    print("="*60)
    print("CONFIGURATION DIMENSIONS:")
    print("="*60)
    
    # Print shape meta from config
    print(f"Shape meta from config:")
    print(f"  Obs shape: {cfg.shape_meta.obs}")
    print(f"  Action shape: {cfg.shape_meta.action}")
    
    # Calculate expected dimensions from config
    obs_total_dim = 0
    for obs_name, obs_spec in cfg.shape_meta.obs.items():
        shape = obs_spec.shape
        if isinstance(shape, (list, tuple, ListConfig)):
            # shape形如 [1024, 3] 或 [9]
            dim = int(shape[-1])          # 取最后一维
        else:
            dim = int(shape)              # 标量
        print(f"    {obs_name}: shape={list(shape) if isinstance(shape,(list,tuple,ListConfig)) else shape}, dim_used={dim}")
        obs_total_dim += dim

    a_shape = cfg.shape_meta.action.shape
    action_dim = int(a_shape[-1]) if isinstance(a_shape, (list, tuple, ListConfig)) else int(a_shape)
    print(f"  Total obs_dim (expected from raw shapes): {obs_total_dim}")
    print(f"  Action dim (expected): {action_dim}")
    
    print("\n" + "="*60)
    print("LOADING DATASET AND MODEL:")
    print("="*60)
    
    # Instantiate the dataset
    dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f"Dataset loaded: {type(dataset).__name__}")
    
    # Get a sample from the dataset
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)  # Small batch for debugging
    sample_batch = next(iter(dataloader))
    
    print(f"\nSample batch keys: {list(sample_batch.keys())}")
    for key, value in sample_batch.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    # Calculate actual dimensions from the batch
    actual_action_dim = sample_batch['action'].shape[-1]
    actual_obs_dim = 0
    if 'obs' in sample_batch:
        for obs_key, obs_tensor in sample_batch['obs'].items():
            obs_dim = obs_tensor.shape[-1]
            actual_obs_dim += obs_dim
            print(f"    Actual {obs_key} dim: {obs_dim}")
    
    print(f"\nActual dimensions from batch:")
    print(f"  Actual action dim: {actual_action_dim}")
    print(f"  Actual obs dim (total): {actual_obs_dim}")
    
    # Instantiate the model
    model: DP3 = hydra.utils.instantiate(cfg.policy)
    print(f"\nModel loaded: {type(model).__name__}")
    
    # Get model's expected dimensions
    print(f"\nModel expected dimensions:")
    print(f"  Model action_dim: {model.action_dim}")
    print(f"  Model obs_dim: {model.obs_dim}")
    print(f"  Expected total: {model.action_dim + model.obs_dim}")
    
    # Test the forward pass to see trajectory construction
    print("\n" + "="*60)
    print("TESTING TRAJECTORY CONSTRUCTION:")
    print("="*60)
    
    try:
        # Move batch to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        batch_on_device = {}
        for key, value in sample_batch.items():
            if isinstance(value, dict):
                batch_on_device[key] = {}
                for subkey, subvalue in value.items():
                    batch_on_device[key][subkey] = subvalue.to(device)
            else:
                batch_on_device[key] = value.to(device)
        
        # Get conditions (this is where trajectory construction happens)
        with torch.no_grad():
            cond = model._build_conditions(batch_on_device['obs'])
            
            print(f"Condition keys: {list(cond.keys())}")
            for key, value in cond.items():
                print(f"  {key}: {value.shape}")
                
            # Calculate total condition dimension
            total_cond_dim = sum([v.shape[-1] for v in cond.values()])
            print(f"  Total condition dim: {total_cond_dim}")
            
            # Check if this matches model's expectation
            print(f"  Model obs_dim: {model.obs_dim}")
            print(f"  Match: {total_cond_dim == model.obs_dim}")
            
            # Get action shape
            action_shape = batch_on_device['action'].shape
            print(f"  Action shape: {action_shape}")
            print(f"  Action dim: {action_shape[-1]}")
            print(f"  Model action_dim: {model.action_dim}")
            print(f"  Match: {action_shape[-1] == model.action_dim}")
            
            # Try to construct trajectory
            B, T = action_shape[:2]
            trajectory = torch.cat([
                batch_on_device['action'],
                cond['0']  # Using first condition key
            ], dim=-1)
            
            print(f"\nTrajectory shape: {trajectory.shape}")
            print(f"Expected trajectory last dim: {model.action_dim + model.obs_dim}")
            print(f"Actual trajectory last dim: {trajectory.shape[-1]}")
            print(f"Match: {trajectory.shape[-1] == model.action_dim + model.obs_dim}")
            
    except Exception as e:
        print(f"Error during trajectory construction: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Config expected obs_dim: {obs_total_dim}")
    print(f"Config expected action_dim: {action_dim}")
    print(f"Actual batch obs_dim: {actual_obs_dim}")
    print(f"Actual batch action_dim: {actual_action_dim}")
    print(f"Model obs_dim: {model.obs_dim}")
    print(f"Model action_dim: {model.action_dim}")
    
    print("\nDimension Mismatches:")
    if obs_total_dim != actual_obs_dim:
        print(f"  Config vs Batch obs_dim: {obs_total_dim} != {actual_obs_dim}")
    if action_dim != actual_action_dim:
        print(f"  Config vs Batch action_dim: {action_dim} != {actual_action_dim}")
    if actual_obs_dim != model.obs_dim:
        print(f"  Batch vs Model obs_dim: {actual_obs_dim} != {model.obs_dim}")
    if actual_action_dim != model.action_dim:
        print(f"  Batch vs Model action_dim: {actual_action_dim} != {model.action_dim}")

if __name__ == "__main__":
    debug_dimensions()