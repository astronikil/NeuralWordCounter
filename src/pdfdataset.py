import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class pdfdataset(Dataset):
    def __init__(self, csvfile, root_dir):
        super().__init__()
        self.pdf_frame = pd.read_csv(root_dir + csvfile)
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.pdf_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        pdf_name = self.root_dir + 'pdf_'+str(idx)+'.csv'

        pdf = torch.tensor([np.loadtxt(pdf_name, dtype='float', delimiter=',')]).float()
        nwords = torch.tensor([self.pdf_frame.iloc[idx, 1]]).float()
        
        sample = {'pdf': pdf, 'nwords': nwords}
        return sample
