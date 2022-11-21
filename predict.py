import json
import torch
import torch.nn.functional as F
from flask_restful import Resource
from utils import load_toxic_model


class ToxicClassifer(Resource):

    def get(self, file):
        with open(file) as f:  # получаем на вход json,в котором содержится наш текст
            json_content = json.load(f)
        text_list = json_content['text']

        model = load_toxic_model(path='state_dict.pt')  # загружаем нашу модель и прогоняем через модель
        logits = []
        probs = []

        with torch.no_grad():
            for string in text_list:
                logits.append(model(string).item())
                probs.append(F.sigmoid(logits).item())

        with open('prediction.json', 'w') as f:  # сохраняем наши предсказания в json file
            json.dump({'logits': logits, 'probabilities': probs}, f)

        return 'prediction.json', 200  # возвращаем названия json в случае успешного выполнения
