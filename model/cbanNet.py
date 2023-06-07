import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTModel, BertModel , RobertaModel
import math

class CbanNet(torch.nn.Module):
    def __init__(self, hidden_dim=768, class_dim=2, dropout_rate=0.1, language='en'):
        super(CbanNet, self).__init__()
        self.class_dim = class_dim

        self.mode = 2
        self.hidden_dim = hidden_dim
        self.language = language

        self.text_encoder = self.get_text_model()
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')

        self.freeze_text_encoder()
        self.freeze_image_encoder()

        self.text_dim = hidden_dim
        self.image_dim = hidden_dim

        self.dropout = nn.Dropout(dropout_rate)

        self.text_project = nn.Linear(self.text_dim, hidden_dim)
        self.image_project = nn.Linear(self.image_dim, hidden_dim)

        if self.mode == 1:
            self.text_fuse = nn.Linear(hidden_dim+hidden_dim, hidden_dim)
            self.image_fuse = nn.Linear(hidden_dim+hidden_dim, hidden_dim)
        elif self.mode == 2:
            self.text_fuse_neg = nn.Linear(hidden_dim+hidden_dim, hidden_dim)
            self.image_fuse_neg = nn.Linear(hidden_dim+hidden_dim, hidden_dim)  
            self.text_fuse = nn.Linear(hidden_dim+hidden_dim, hidden_dim)
            self.image_fuse = nn.Linear(hidden_dim+hidden_dim, hidden_dim)  

        self.t_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.i_weight = nn.Linear(hidden_dim, 1, bias=False)

        self.fc = nn.Linear(2*hidden_dim, self.class_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh() 
        self.relu = nn.ReLU()

    def get_text_model(self):
        if self.language == 'en':
            return AutoModel.from_pretrained("bert-base-uncased")
        elif self.language == 'cn':
            return AutoModel.from_pretrained("bert-base-chinese")

    def freeze_text_encoder(self):
        for name, param in list(self.text_encoder.named_parameters()):
            if self.language == 'en':
                if 'pooler' in name or 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'encoder.layer.9' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            elif self.language == 'cn':
                if 'pooler' in name or 'encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def freeze_image_encoder(self):
        for name, param in list(self.image_encoder.named_parameters()):
            if name.startswith('pooler') or 'encoder.layer.11' in name:
                continue
            else:
                param.requires_grad = False

    def attention_net(self, x, weight):
        d_k = x.size(-1)
        if weight == 't':
            scores = self.t_weight(self.dropout(x))/math.sqrt(d_k) 
        elif weight =='i':
            scores = self.i_weight(self.dropout(x))/math.sqrt(d_k)
        scores = scores.transpose(1,2)
        p_attn = F.softmax(scores,-1)
        context = torch.bmm(p_attn, x).squeeze(1)
        return context, p_attn

    def forward(self, input_ids, attention_mask, images, image_mask):

        t_feature = self.text_encoder(input_ids=input_ids,attention_mask=attention_mask)
        i_feature = self.image_encoder(images)

        t_feature = t_feature[0]
        i_feature = i_feature[0]

        if self.mode == 0:
            t_final = t_feature
            i_final = i_feature

        elif self.mode == 1 or self.mode == 2:
            t_p = self.tanh(self.text_project(self.dropout(t_feature)))
            i_p = self.tanh(self.image_project(self.dropout(i_feature)))
            atten_map = torch.bmm(t_p,i_p.transpose(1,2))
            scores_t = F.softmax(atten_map, dim = -1)
            attened_t = torch.bmm(scores_t,i_feature)

            if self.mode == 2:
                atten_map_neg = torch.mul(atten_map,-1)
                scores_t_neg = F.softmax(atten_map_neg, dim = -1)
                attened_t_neg = torch.bmm(scores_t_neg,i_feature)
            
            atten_map_t = atten_map.transpose(1,2)
            scores_i = F.softmax(atten_map_t, dim = -1)
            attened_i = torch.bmm(scores_i,t_feature)

            if self.mode == 2:
                atten_map_t_neg = atten_map_neg.transpose(1,2)
                scores_i_neg = F.softmax(atten_map_t_neg, dim = -1)
                attened_i_neg = torch.bmm(scores_i_neg,t_feature)

            if self.mode == 1:
                t_final = torch.cat([t_feature,attened_t], dim=-1)
            elif self.mode == 2:
                t_fuse = torch.cat([attened_t,attened_t_neg], dim=-1)
                t_fuse = self.tanh(self.text_fuse_neg(self.dropout(t_fuse)))
                t_final = torch.cat([t_feature,t_fuse], dim=-1)

            t_final = self.tanh(self.text_fuse(self.dropout(t_final)))

            if self.mode == 1:        
                i_final = torch.cat([i_feature,attened_i], dim=-1)
            elif self.mode == 2:
                i_fuse = torch.cat([attened_i,attened_i_neg], dim=-1)
                i_fuse = self.tanh(self.image_fuse_neg(self.dropout(i_fuse)))
                i_final = torch.cat([i_feature,i_fuse], dim=-1)

            i_final = self.tanh(self.image_fuse(self.dropout(i_final)))

        t_final, t_a = self.attention_net(t_final,'t')
        i_final, i_a = self.attention_net(i_final,'i')

        final = torch.cat([t_final,i_final], dim=-1)
    
        x = self.fc(self.dropout(final))
        
        return x