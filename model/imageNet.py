import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel, BertModel

class ImageNet(nn.Module):
    def __init__(self,  hidden_dim=768, class_dim=2, dropout_rate=0.5, language='en'):

        super(ImageNet, self).__init__()

        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.language = language

        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')

        self.freeze_image_encoder()

        self.dropout = nn.Dropout(self.dropout_rate)
  
        self.fc = nn.Linear(self.hidden_dim, self.class_dim)

    def freeze_image_encoder(self):
        for name, param in list(self.image_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name:
                continue
            else:
                param.requires_grad = False

    def forward(self, images, image_mask):

        i = self.image_encoder(images)
        i = i['pooler_output']
        
        f = self.dropout(i)

        return self.fc(f)