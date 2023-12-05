from Data import *
import torch
import numpy as np
import time
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup, AdamW


train_list = rest_train
test_list = rest_trial
test_sent = set([item[0] for item in test_list])
test_index = {sent:[] for sent in test_sent}
for n, item in enumerate(test_list):  # Build sent-index map in order to get test accuracy
    test_index[item[0]].append(n)
train_sent = set([item[0] for item in train_list])
val_sent = set(random.sample(list(train_sent), int(0.2*len(train_sent))))  # Train valid split
val_list = [item for item in train_list if item[0] in val_sent]
train_list = [item for item in train_list if item[0] not in val_sent]


train_set = ABSASet(train_list)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_set = ABSASet(val_list)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_set = ABSASet(val_list)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


num_labels = 3
num_epochs = 5
total_steps = len(train_loader) * num_epochs
learning_rate = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

roberta = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
).to(device)


def evaluate(model):
    model.eval()
    total_loss = 0
    for token, label in val_loader:
        token = token.to(device)
        label = label.to(device).flatten()
        output = model(**token, labels=label)
        total_loss += output.loss.item()
    return total_loss / len(val_loader)


def train(model, epochs):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for i in range(epochs):
        print('========= Epoch {} / {} =========='.format(i+1, epochs))
        t0 = time.time()
        total_train_loss = 0
        for step, (token, label) in enumerate(train_loader):
            if step % 40 == 0 and not step == 0:
                elapsed = round(time.time() - t0, 2)
                print('Batch {} of {}. Elapsed: {}.'.format(step, len(train_loader), elapsed))
            token = token.to(device)
            label = label.to(device).flatten()
            optimizer.zero_grad()
            loss = model(**token, labels=label).loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        training_time = round(time.time()-t0, 2)
        print("Average training loss: {}".format(avg_train_loss))
        print("Training epoch took: {}".format(training_time))
        val_loss, f1, acc = evaluate(model)
        print("Val loss: {}".format(val_loss))
    print("Training complete!")


def test(model):
    model.eval()
    right = [0 for _ in range(len(test_loader))]
    for i, (token, label) in enumerate(test_loader):
        token = token.to(device)
        label = label.to(device).flatten()
        logits = model(**token).logits
        if logits.argmax().item() == label.item():
            right[i] = 1
    acc = 0
    for sent in test_sent:
        for i in test_index[sent]:
            if right[i] == 0:
                continue
        acc += 1
    return acc/len(test_sent)


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train(roberta, num_epochs)
print("Test acc:", test(roberta))
