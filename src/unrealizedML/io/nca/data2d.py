import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
from pathlib import Path

import re

def pickle_dir(in_dir: Path, chunksize: int, out: Path):
    
    file_list = [ file.resolve() for file in in_dir.iterdir() ]
    
    chunk = 0
    chunk_data = None
    
    print(file_list)
    
    chunks = int(np.round(len(file_list) / chunksize))
    
    print(chunks)
    
    for chunk in range(chunks):
        
        chunk_data = np.array([])
        
        index = chunk * chunksize
        
        for file in file_list[index:index + chunksize]:
            
            img = Image.open(file)
            img = img.convert('RGB')
            
            current_data = np.array(img)
            
            reshape_shape = (1, current_data.shape[-1], current_data.shape[0], current_data.shape[1])
            current_data = current_data.reshape(reshape_shape)
            
            
            if not chunk_data.any():
                
                chunk_data = current_data
                continue
            
            chunk_data = np.concatenate([ chunk_data, current_data], 0)
        
        chunk_data = torch.tensor(chunk_data, dtype=torch.uint8)
        torch.save(chunk_data, f'{out.resolve()}/{chunk}.pkl')
    
class TrainingData2d(Dataset):
    
    def __init__(self, data_in: Path, data_out: Path):
        super().__init__()
        
        data_in_files = [ file.resolve() for file in data_in.iterdir() ]
        data_out_files = [ file.resolve() for file in data_out.iterdir() ]
        
        self.data_in_map = self.create_mapping(data_in_files)
        self.data_out_map = self.create_mapping(data_out_files)
        
        assert len(self.data_in_map) == len(self.data_out_map), 'data in and data out mapping mismatch'
        assert self.data_in_map.keys() == self.data_out_map.keys(), 'data in keys and data out keys mismatch'
        
    def create_mapping(self, file_list: list[Path]):
        
        pairs = []
        
        for file in file_list:
            
            is_valid = re.search('^\d+\.pkl$', file.name)
            
            if is_valid:
                
                chunk_num = int(re.sub('\.pkl', '', file.name))
                
                pairs.append((chunk_num, file))
        
        mapping = dict()
        mapping.update(pairs)
        
        return mapping
    
    def __len__(self):
        
        return len(self.data_in_map)
    
    def __getitem__(self, index):
        
        in_data = torch.load(self.data_in_map[index])
        out_data = torch.load(self.data_out_map[index])
        
        return (in_data, out_data)