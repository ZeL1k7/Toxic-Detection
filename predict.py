from functools import lru_cache
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from flask_restful import Resource
from utils import load_toxic_model


class ToxicClassifer(Resource):
    def __init__(self):
        self.model = self.make_pt_model('state_dict.pt')
        self.output_filepath = 'predictions.json'

    @lru_cache(1)
    def make_pt_model(self, model_path: Path) -> torch.nn.Module:
        """
        Загрузка и кэширование предобученной модели
        :param model_path: путь до state_dict предобученной модели
        :return: torch.nn.Module: предобученная модель
        """
        print('Started loading model')
        model = load_toxic_model(model_path)
        print('Successful loaded model')
        return model

    def predict_from_json(self, path: Path) -> Path:
        """
        Inference модели
        :param path: путь до json с исходными данными
        :return: output_filepath: путь до json с результатами модели
        """
        # получаем на вход json,в котором содержится наш текст
        with open(path) as file:
            data = json.load(file)
        text_list = data['text']

        logits = []
        probs = []

        with torch.no_grad():
            for text in text_list:
                logits.append(self.model(text).item())
                probs.append(F.sigmoid(logits).item())
        # сохраняем наши предсказания в json file
        with open(self.output_filepath, 'w+') as file:
            json.dump({'logits': logits, 'probabilities': probs}, file)
        return self.output_filepath
