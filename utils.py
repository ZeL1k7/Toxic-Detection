import torch
from pathlib import Path
from torch.utils.data import Dataset
from model import ToxicModel


class ToxicDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.x = df['comment'].values
        self.y = df['toxic'].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple(str, float):
        return self.x[idx], self.y[idx]


def load_toxic_model(path: Path = 'state_dict.pt', mode: str = 'eval') -> torch.nn.Module:
    """
    Загружает предобученную модель используя state_dict и переводит в нужный режим
    :param  path: Path: указывает путь до state_dict модели
    :param mode: str: режимы модели (eval - для inference, train - для обучения)
    :return:  torch.nn.Module: возвращает предобученную модель
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ToxicModel()
    model.load_state_dict(torch.load(path, map_location=torch.device(device=device)))
    if mode == 'eval':
        model.eval()
    elif model == 'train':
        model.train()
    return model
