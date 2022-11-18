import torch
import torch.nn.functional as F
from utils import load_toxic_model
import json


def get_prediction(text):
    model = load_toxic_model(path='state_dict.pt')
    with torch.no_grad():
        logits = model(text)
        probs = F.sigmoid(logits)
    with open('prediction.json', 'w') as f:
        json.dump({'logits': logits.item(), 'probalities': probs.item()}, f)
    return 'prediction.json'
