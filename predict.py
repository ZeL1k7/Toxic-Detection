from functools import lru_cache
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from utils import load_toxic_model, DHExchange


class ToxicClassifer:
    def __init__(self):
        self.model = self.make_pt_model('state_dict.pt')
        self.output_filepath = 'predictions.json'
        c_public = 197
        s_public = 151
        s_private = 157
        self.server = DHExchange(c_public, s_public, s_private)
        #s_full = server.generate_full_key(s_partial)

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

    def get_client_partial_key(self, s_partial):
        return self.server.generate_partial_key()


    def predict_from_json(self, c_partial, path: Path) -> Path:
        """
        Inference модели
        :param path: путь до json с исходными данными
        :return: output_filepath: путь до json с результатами модели
        """
        # получаем на вход json,в котором содержится наш текст

        s_full = self.server.generate_full_key(c_partial)
        path = self.server.decrypt_message(path)

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
        return  self.output_filepath
