import torch
import torch.nn as nn
from transformers import BertModel, ViTModel, AutoModel , RobertaModel
from transformers.adapters import AdapterConfig , LoRAConfig

class TextNet(nn.Module):
    def __init__(self,  hidden_dim=768, class_dim=2, dropout_rate=0.1, language='en'):

        super(TextNet, self).__init__()

        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language

        self.text_encoder = self.get_text_model()

        self.freeze_text_encoder()

        self.dropout = nn.Dropout(self.dropout_rate)
  
        self.fc = nn.Linear(self.hidden_dim, self.class_dim)

    def get_text_model(self):
        if self.language == 'en':
            return AutoModel.from_pretrained("bert-base-uncased")
        elif self.language == 'cn':
            return AutoModel.from_pretrained("bert-base-chinese")

    def freeze_text_encoder(self):
        for name, param in list(self.text_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name:
                continue
            else:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
        x = x[1]

        f = self.dropout(x)

        return self.fc(f)