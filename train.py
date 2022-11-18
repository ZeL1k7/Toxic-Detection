import gc

import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm

from utils import ToxicDataset, load_toxic_model


if __name__ == "__main__":
    df = pd.read_csv('labeled.csv')
    model = load_toxic_model()
    device = model.device
    model.to(device)
    dataset = ToxicDataset(df)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), 2e-4)
    writer = SummaryWriter()
    for epoch in range(15):
        for x, y in tqdm(loader):
            y = y.unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.cpu(), y)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch)
            del x,y
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    torch.save(model.state_dict(), 'state_dict.pt')
