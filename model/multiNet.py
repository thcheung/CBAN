import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel, BertModel , RobertaModel

class MultiNet(nn.Module):
    def __init__(self,  hidden_dim=768, class_dim=2, dropout_rate=0.5, language='en'):

        super(MultiNet, self).__init__()

        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language

        self.text_encoder = self.get_text_model()
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')

        self.freeze_text_encoder()
        self.freeze_image_encoder()

        self.dropout = nn.Dropout(self.dropout_rate)
  
        self.fc = nn.Linear(self.hidden_dim + self.hidden_dim, self.class_dim)

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

    def freeze_image_encoder(self):
        for name, param in list(self.image_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name:
                continue
            else:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, images, image_mask):

        x = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        i = self.image_encoder(images)

        x = x['pooler_output']
        i = i['pooler_output']
        
        f = torch.cat([x,i],dim=-1)

        f = self.dropout(f)

        return self.fc(f)