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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ToxicModel()
    model.load_state_dict(torch.load(path, map_location=torch.device(device=device)))
    model.eval()
    return model
