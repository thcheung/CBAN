import argparse
from ast import Mult
import torch
import random
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import MultiDataset
from model import get_model
from utils import print_metrics
from experiment import get_experiment
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings("ignore")

RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Multimodal Rumor Detection and Verification')

parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 16)')

parser.add_argument('--epoch', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')

parser.add_argument('--hidden_dim', type=int, default=768, metavar='N',
                    help='hidden dimension (default: 768)')

parser.add_argument('--max_len', type=int, default=64, metavar='N',
                    help='maximum length of the conversation (default: 32)')

parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                    help='dropout rate (default: 0.5)')

parser.add_argument('--model', type=str, default="cban", metavar='N',
                    help='model name')

parser.add_argument('--experiment', type=str, metavar='N',
                    help='experiment name')

parser.add_argument('--fold', type=int, default=0,  metavar='N',
                    help='experiment name')

args = parser.parse_args()


def train():

    experiment = get_experiment(args.experiment)

    image_dir = experiment["image_dir"]

    root_dir = os.path.join(experiment["root_dir"], str(args.fold))

    language = experiment["language"]

    classes = experiment["classes"]

    test_path = os.path.join(root_dir, "test.json")

    test_dataset = MultiDataset(
        test_path, image_dir, classes, train=False, language=language)

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    model = get_model(args.model,args.hidden_dim, len(classes),
                         args.dropout, language=language)

    model = nn.DataParallel(model)

    model = model.to(device)

    comment = f'{args.model}_{args.experiment}_{args.fold}'

    checkpoint_dir = os.path.join("checkpoints/",comment)

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    model.module.load_state_dict(torch.load(checkpoint_path))

    model.eval()

    test_count = 0
    test_predicts = []
    test_labels = []

    for i, batch in enumerate(tqdm(test_dataloader)):

        images = batch["image"].to(device)
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['label'].to(device)
        image_mask = batch['image_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, images=images, image_mask=image_mask)

        _, preds = torch.max(outputs, 1)
    
        test_count += labels.size(0)

        for pred in preds.tolist():
            test_predicts.append(pred)
        for lab in labels.tolist():
            test_labels.append(lab)

    all_labels = [int(test_label) for test_label in test_labels]

    all_labels = list(set(all_labels))
    all_labels = sorted(all_labels)

    print_metrics(test_labels, test_predicts)
    print(classification_report(test_labels, test_predicts,labels=all_labels,digits=3))

if __name__ == "__main__":
    train()
