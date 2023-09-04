import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import random

batch_size = 8
num_workers = 0  # Change based on the server

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
mask = "[MASK]"
sep = "[SEP]"


class ABSASet(Dataset):
    def __init__(self, data, mask_rate=0):
        self.data = data
        self.mask_rate = mask_rate

    def __getitem__(self, index):
        sent, aspect, polarity = self.data[index]
        if random.random() < self.mask_rate:
            pair = sent.replace(aspect, mask) + sep + mask
        else:
            pair = sent + sep + aspect
        encoding = tokenizer(pair, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        encoding["input_ids"].squeeze_(0)
        encoding["token_type_ids"].squeeze_(0)
        encoding["attention_mask"].squeeze_(0)
        return encoding, torch.Tensor([polarity]).long()

    def __len__(self):
        return len(self.data)


with open("../Laptops_Test.pkl", "rb") as f:
    laptop_trial = pickle.load(f)

with open("../Laptops_Train.pkl", "rb") as f:
    laptop_train = pickle.load(f)

with open("../Restaurants_Test.pkl", "rb") as f:
    rest_trial = pickle.load(f)

with open("../Restaurants_Train.pkl", "rb") as f:
    rest_train = pickle.load(f)

with open("../Mixed_Test.pkl", "rb") as f:
    mix_trial = pickle.load(f)

with open("../Mixed_Train.pkl", "rb") as f:
    mix_train = pickle.load(f)


# laptop_trial_set = ABSASet(laptop_trial)
# laptop_trial_loader = DataLoader(laptop_trial_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
# rest_trial_set = ABSASet(rest_trial)
# rest_trial_loader = DataLoader(rest_trial_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
# mix_trial_set = ABSASet(mix_trial)
# mix_trial_loader = DataLoader(mix_trial_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#
# laptop_train_set = ABSASet(laptop_train)
# laptop_train_loader = DataLoader(laptop_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
# rest_train_set = ABSASet(rest_train)
# rest_train_loader = DataLoader(rest_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
# mix_train_set = ABSASet(mix_train)
# mix_train_loader = DataLoader(mix_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
