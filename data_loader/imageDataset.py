import json
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import preprocess, image_transforms
from PIL import Image, ImageFile
from transformers import BertTokenizer, ViTFeatureExtractor , AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, label_path, image_dir, classes, train=False, max_length=64, language='en'):
        self.train = train
        self.label_path = label_path
        self.image_dir = image_dir
        self.max_length = max_length
        self.language = language
        self.classes = classes
        self.image_transforms = image_transforms()
        
        self.text_tokenizer = self._get_text_tokenizer()
        self.image_tokenizer = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

        self.ids, self.texts, self.labels, self.images = self.process_file()

    def _get_text_tokenizer(self):
        if self.language == 'en':
            return AutoTokenizer.from_pretrained("bert-base-uncased")
        elif self.language == 'cn':
            return AutoTokenizer.from_pretrained("bert-base-chinese")

    def tensor_to_label(self, tensor):
        tensor = int(tensor)
        return self.classes[tensor]

    def label_to_tensor(self, label):
        index = self.classes.index(label)
        return torch.tensor(index)

    def _get_text(self, text):
        text = preprocess(text)
        return self.text_tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True,return_tensors='pt')

    def _get_image(self, image_id):
        return os.path.join(self.image_dir, image_id)

    def _get_labels(self, label):
        return self.label_to_tensor(label)

    def process_file(self):
        ids = []
        texts = []
        labels = []
        images = []

        with open(self.label_path, 'r', encoding='utf-8') as f:
            tweets = json.load(f)
            
            for tweet in tqdm(tweets):
                id = tweet['id']
            
                if "image" not in tweet.keys() or tweet["image"] == None or tweet["image"] == '':
                    image_path = "blank_image.jpg"
                else:
                    image_path = self._get_image(tweet["image"])
        
                image = Image.open(image_path).convert("RGB")
                
                image = self.image_transforms(image)

                text = self._get_text(tweet['text'])
                label = self._get_labels(tweet['label'])
                ids.append(id)
                texts.append(text)
                labels.append(label)
                images.append(image)

        return ids, texts, labels, images

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        item = self.texts[idx]

        image = self.images[idx]
        image = self.image_tokenizer(image, return_tensors="pt")

        id = self.ids[idx]
        label = self.labels[idx]

        item['image']  = image['pixel_values'][0]
        item['image_mask'] = torch.zeros(1)
        
        item['id'] = id
        item['label'] = label
        # item['image_path'] = image_path
        return item
