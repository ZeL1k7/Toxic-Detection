import json
import torch
import torch.nn.functional as F
from flask_restful import Resource
from functools import lru_cache
from utils import load_toxic_model

class ToxicClassifer(Resource):

    def __init__(self,model):
        self.model = self.make_pt_model('state_dict.pt')
        self.output_filepath = 'predictions.json'
    @lru_cache(1)  # загружаем модель + кэшируем ее
    def make_pt_model(self,model_path):
        print('Started loading model')
        model = load_toxic_model(model_path)
        print('Successful loaded model')
        return model

    def get(self, filepath):

        try:
            with open (filepath) as file:  # получаем на вход json,в котором содержится наш текст
                data = json.load(file)
            text_list = data['text']

            logits = []
            probs = []

            with torch.no_grad():
                for text in text_list:
                    logits.append(self.model(text).item())
                    probs.append(F.sigmoid(logits).item())

            with open(self.output_filepath, 'w+') as f:  # сохраняем наши предсказания в json file
                json.dump({'logits': logits, 'probabilities': probs}, f)

            return self.output_filepath, 200  # возвращаем названия json в случае успешного выполнения
        except:
            'error', 400

'''
TO DO!!!

from pedantic import BaseModel

class Item(BaseModel):
id: int = Field(..., description="Unique item id")
text: str = Field(..., description="какой то текст который ты там передаешь?")

'''