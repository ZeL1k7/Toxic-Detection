import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class ToxicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.classifier = nn.Linear(768, 1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, X):
        with torch.no_grad():
            encoded = self.tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt')
            output = self.bert(**encoded)
        embedding = self.mean_pooling(output, encoded['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return self.classifier(embedding)
