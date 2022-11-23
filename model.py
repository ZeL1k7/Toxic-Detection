import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class ToxicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.classifier = nn.Linear(768, 1)

    def mean_pooling(self, model_output: tuple(torch.FloatTensor), attention_mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        Усредняет все вектора эмбеддингов в один вектор
        :param model_output: tuple(torch.FloatTensor)
        :param attention_mask: torch.FloatTensor
        :return: torch.FloatTensor: усредненный вектор
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            encoded = self.tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            output = self.bert(**encoded)
        embedding = self.mean_pooling(output, encoded['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1).to(self.device)
        return self.classifier(embedding)
