import csv
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data.distributed import DistributedSampler

from utils import get_event_cond


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths, params, labels):
        super().__init__()
        self.filenames = []
        self.audio_length = params['audio_length']
        self.labels = labels
        self.event_type = params['event_type']
        for path in paths:
            self.filenames += self.parse_filelist(path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        signal = signal[0, :self.audio_length]
            
        # extract class cond
        cls_name = os.path.dirname(audio_filename).split('/')[-1]
        cls = torch.tensor(self.labels.index(cls_name))
        
        # extract event cond
        event = signal.clone().detach()
        event = get_event_cond(event, self.event_type)
        
        return {
            'audio': signal,
            'class': cls,
            'event': event
        }
        
    def parse_filelist(self, filelist_path):
        # if filelist_path is txt file
        if filelist_path.endswith('.txt'):
            with open(filelist_path, 'r') as f:
                filelist = [line.strip() for line in f.readlines()]
            return filelist
        
        # if filelist_path is csv file
        if filelist_path.endswith('.csv'):
            with open(filelist_path, 'r') as f:
                reader = csv.reader(f)
                filelist = [row[0] for row in reader]
                f.close()
            return filelist
    
    def moving_avg(self, input, window_size):
        if type(input) != list: input = list(input)
        result = []
        for i in range(1, window_size+1):
            result.append(sum(input[:i])/i)
        
        moving_sum = sum(input[:window_size])
        result.append(moving_sum/window_size)
        for i in range(len(input) - window_size):
            moving_sum += (input[i+window_size] - input[i])
            result.append(moving_sum/window_size)
        return np.array(result)
    
    

def from_path(data_dirs, params, labels, distributed=False):
    dataset = AudioDataset(data_dirs, params, labels)
    if distributed:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=params['batch_size'],
            collate_fn=None,
            shuffle=False,
            num_workers=params['num_workers'],
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(dataset))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        collate_fn=None,
        shuffle=True,
        num_workers=os.cpu_count()//4,
        pin_memory=True,
        drop_last=True)
