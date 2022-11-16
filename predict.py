import torch
import torch.nn.functional as F
from utils import load_toxic_model


def get_prediction(text):
    model = load_toxic_model(path='state_dict.pt')
    with torch.no_grad():
        logits = model(text)
        probs = F.sigmoid(logits)
    return probs.item()
