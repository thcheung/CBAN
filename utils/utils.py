import re
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        # t = '' if t.startswith('#') and len(t) > 1 else t
        t = '' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def print_metrics(y_true, y_pred):
    print(f"Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro')}")
    print(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro')}")
    print(
        f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted')}")
    print(f"Accuracy): {accuracy_score(y_true, y_pred)}")


def labels_to_weights(labels):
    num = max(labels) + 1
    counts = [labels.count(i) for i in range(0, num)]
    total = sum(counts)
    counts = [total/count for count in counts]
    return torch.tensor(counts, dtype=torch.float)
    
def image_transforms():
    transforms = []
    transforms.append(T.Resize(384))
    transforms.append(T.CenterCrop(384))
    return T.Compose(transforms)

