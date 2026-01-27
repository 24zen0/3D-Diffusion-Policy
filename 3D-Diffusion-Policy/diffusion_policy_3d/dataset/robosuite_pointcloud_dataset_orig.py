from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RobosuitePointcloudDataset(BaseDataset):
    """
    Dataset class for loading and processing Robosuite point cloud data.
    
    This dataset handles point cloud observations, agent positions, and actions
    from Robosuite simulation environments stored in zarr format.
    """
    
    def __init__(self,
            zarr_path: str,  # Path to the zarr data file
            horizon: int = 1,  # Number of consecutive timesteps in each sample
            pad_before: int = 0,  # Padding at the beginning of sequences
            pad_after: int = 0,  # Padding at the end of sequences
            seed: int = 42,  # Random seed for reproducibility
            val_ratio: float = 0.0,  # Ratio of episodes to use for validation
            max_train_episodes: int = None,  # Maximum number of episodes for training
            ):
        super().__init__()
        
        # Load data from zarr file, extracting agent_pos, action, and point_cloud keys
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['agent_pos', 'action', 'point_cloud'])
        
        # Create validation mask to split data into training and validation sets
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        
        # Training mask is the complement of validation mask
        train_mask = ~val_mask
        
        # Optionally downsample the training episodes
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # Initialize sampler for generating sequences from the replay buffer
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        # Store configuration parameters
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """
        Create a separate dataset instance for validation using the validation episodes.
        
        Returns:
            RobosuitePointcloudDataset: A new dataset instance for validation
        """
        val_set = copy.copy(self)
        
        # Create sampler for validation episodes (complement of training mask)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = 'limits', **kwargs):
        """
        Compute normalization statistics for the dataset.
        
        Args:
            mode: Normalization mode ('limits', 'std', etc.)
            **kwargs: Additional arguments for normalization
            
        Returns:
            LinearNormalizer: Normalizer object with computed statistics
        """
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['agent_pos'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        
        # Create and fit normalizer to the data
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Note: Point cloud normalization is commented out (uses identity transformation)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        # normalizer['point_cloud_robot'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        """Return the total number of sequences in the dataset."""
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict) -> Dict:
        """
        Convert a raw sample from the replay buffer to the training data format.
        
        Args:
            sample: Dictionary containing raw sample data
            
        Returns:
            Dictionary with processed observation and action data
        """
        # Extract agent position: (T, D_state=7) where:
        # ee_pos: 3, ee_rotvec: 3, gripper_gap: 1
        agent_pos = sample['agent_pos'][:,].astype(np.float32)
        
        # Extract point cloud: (T, 1024, 3) - 1024 points with x,y,z coordinates
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        # Create structured data dictionary
        data = {
            'obs': {
                'point_cloud': point_cloud,  # Shape: (T, 1024, 3)
                'agent_pos': agent_pos,      # Shape: (T, D_pos=7)
            },
            'action': sample['action'].astype(np.float32)  # Shape: (T, D_action=7)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample by index.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing torch tensors for observations and actions
        """
        # Sample a sequence from the replay buffer
        sample = self.sampler.sample_sequence(idx)
        
        # Convert to structured data format
        data = self._sample_to_data(sample)
        
        # Convert numpy arrays to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)
        
        return torch_data