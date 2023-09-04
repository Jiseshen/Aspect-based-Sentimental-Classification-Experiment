import torch
import pickle
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import random

batch_size = 1  # MRC
num_workers = 0  # Change based on the server

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


with open('../laptop_train.json', 'r') as f:
    train_sent = json.load(f)

with open('../laptop_test.json', 'r') as f:
    test_sent = json.load(f)


val_sent = random.sample(list(train_sent), int(0.2*len(train_sent)))  # Train valid split
train_sent = [sent for sent in train_sent if sent not in val_sent]



