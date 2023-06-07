import argparse
import time
import torch
import random
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from data_loader import TextDataset
from model import get_model
from utils import print_metrics, labels_to_weights
from experiment import get_experiment
from transformers import logging
import warnings
from transformers import get_linear_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings("ignore")

RANDOM_SEED = 1

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Multimodal Rumor Detection and Verification')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=3e-5, metavar='N',
                    help='learning rate (default: 2e-5)')
                    
parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='N',
                    help='weight decay (default: 1e-3)')

parser.add_argument('--hidden_dim', type=int, default=768, metavar='N',
                    help='hidden dimension (default: 768)')

parser.add_argument('--max_len', type=int, default=64, metavar='N',
                    help='maximum length of the conversation (default: 64)')

parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                    help='dropout rate (default: 0.1)')

parser.add_argument('--experiment', type=str, metavar='N',
                    help='experiment name')

parser.add_argument('--model', type=str, default="text", metavar='N',
                    help='model name')

parser.add_argument('--fold', type=int, default=0,  metavar='N',
                    help='experiment name')

args = parser.parse_args()

def train():

    experiment = get_experiment(args.experiment)

    image_dir = experiment["image_dir"]

    root_dir = os.path.join(experiment["root_dir"], str(args.fold))

    language = experiment["language"]

    classes = experiment["classes"]

    train_path = os.path.join(root_dir, "train.json")

    train_dataset = TextDataset(
        train_path, image_dir, classes, train=True, language=language)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_path = os.path.join(root_dir, "test.json")

    test_dataset = TextDataset(
        test_path, image_dir, classes, train=False, language=language)

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    print('num of training / testing samples : {} / {} '.format(len(train_dataset), len(test_dataset)))

    model = get_model(args.model,args.hidden_dim, len(classes),
                         args.dropout, language=language)

    model = nn.DataParallel(model)

    model = model.to(device)

    labels = [int(item) for item in train_dataset.labels]
    weight = labels_to_weights(labels).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer,int(0.1*args.epochs),int(args.epochs))

    comment = f'{args.model}_{args.experiment}_{args.fold}'

    writer = SummaryWriter(log_dir="runs/{}_{}".format(str(int(time.time())),"train_" + comment))

    checkpoint_dir = os.path.join("checkpoints/",comment)
    
    os.makedirs(checkpoint_dir,exist_ok=True)
    
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs+1):

        model.train()
        train_loss = 0.0
        train_count = 0 

        train_predicts = []
        train_labels = []

        for i, batch in enumerate(tqdm(train_dataloader)):

            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)

            train_loss += loss.item() * labels.size(0)

            loss.backward()

            _, preds = torch.max(outputs, 1)

            train_count += labels.size(0)

            for pred in preds.tolist():
                train_predicts.append(pred)

            for lab in labels.tolist():
                train_labels.append(lab)

            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        train_loss = train_loss/train_count
        train_acc = accuracy_score(train_labels, train_predicts)
        train_f1 = f1_score(train_labels, train_predicts, average='macro')

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("F1_MAC/train", train_f1, epoch)

        model.eval()

        test_loss = 0.0
        test_count = 0

        test_predicts = []
        test_labels = []

        for i, batch in enumerate(tqdm(test_dataloader)):

            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                loss =  criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)

            _, preds = torch.max(outputs, 1)

            test_count += labels.size(0)

            for pred in preds.tolist():
                test_predicts.append(pred)

            for lab in labels.tolist():
                test_labels.append(lab)


        test_loss = test_loss/test_count
        test_acc = accuracy_score(test_labels, test_predicts)
        test_f1 = f1_score(test_labels, test_predicts, average='macro')

        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("F1_MAC/test", test_f1, epoch)

        print("Epoch: {} / {}".format(epoch, args.epochs))
        print_metrics(test_labels, test_predicts)
        
        if test_f1 > best_f1:
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.module.state_dict(),checkpoint_path)
            best_f1 = test_f1

if __name__ == "__main__":
    train()
