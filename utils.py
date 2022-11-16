import torch
from torch.utils.data import Dataset

from model import ToxicModel


class ToxicDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.x = df['comment'].values
        self.y = df['toxic'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
def load_toxic_model(path='state_dict.pt'):
    model = ToxicModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
