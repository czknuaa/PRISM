from typing import Dict
import torch
import numpy as np
import copy
from prism.common.pytorch_util import dict_apply
from prism.common.replay_buffer import ReplayBuffer
from prism.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from prism.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from prism.dataset.base_dataset import BaseDataset
import pdb
import os
from termcolor import cprint
class RobotDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,):
        super().__init__()
        self.task_name = task_name
        
        if os.path.isdir(os.path.join(zarr_path, "data/point_label")):
            keys = ['state', 'action', 'point_cloud','point_label']
            cprint("is_dataset_use_dbscan: Ture","green")
        else:
            keys = ['state', 'action', 'point_cloud']
            cprint("is_dataset_use_dbscan: False","yellow")

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys = keys) # 'img' #point label
      
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):    #此处mode 为limits
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
       
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        if 'point_label' in sample:
            point_label = sample['point_label'][:,].astype(np.int64)     #point label
            data = {
                'obs': {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'point_label': point_label, 
                },
                'action': sample['action'].astype(np.float32) # T, D_action
            }
        else:
            data = {
                'obs': {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                },
                'action': sample['action'].astype(np.float32) # T, D_action
            }    
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == '__main__':
    test = RobotDataset('../../data/mug_hanging_easy_L515_1.zarr')
    print('ready')