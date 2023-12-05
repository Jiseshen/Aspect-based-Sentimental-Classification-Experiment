from transformers import get_linear_schedule_with_warmup, AdamW
import time
from sklearn.metrics import f1_score2E

from Model import *
from Data import *

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 4
num_epochs = 10
total_steps = len(train_set)
learning_rate = 1e-3
criterion = torch.nn.CrossEntropyLoss()


def evaluate(model):
    model.eval()
    total_loss = 0
    for data, s, e, label in test_set:
        data = data.to(device)
        label = label.to(device).flatten()
        logit = model(data, range(s, e)).unsqueeze(0)
        total_loss += criterion(logit, label).item()
    return total_loss / len(test_set)


def train(model, epochs):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_epochs * total_steps)
    for i in range(epochs):
        print('========= Epoch {} / {} =========='.format(i+1, epochs))
        t0 = time.time()
        total_train_loss = 0
        for step, (data, s, e, label) in enumerate(train_set):
            if step % 40 == 0 and not step == 0:
                elapsed = round(time.time() - t0, 2)
                print('Batch {} of {}. Elapsed: {}.'.format(step, len(train_set), elapsed))
            data = data.to(device)
            label = label.to(device).flatten()
            optimizer.zero_grad()
            logit = model(data, range(s, e)).unsqueeze(0)
            loss = criterion(logit, label)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_set)
        training_time = round(time.time()-t0, 2)
        print("Average training loss: {}".format(avg_train_loss))
        print("Training epoch took: {}".format(training_time))
        val_loss = evaluate(model)
        print("Val loss: {}".format(val_loss))
    print("Training complete!")


def test(model):
    model.eval()
    right = [0 for _ in range(len(test_set))]
    predicted = []
    true = []
    for i, (data, s, e, label) in enumerate(test_set):
        data = data.to(device)
        label = label.to(device).flatten()
        logits = model(data, range(s, e))
        predicted.append(logits.argmax().item())
        true.append(label.item())
        if logits.argmax().item() == label.item():
            right[i] = 1
    acc = 0
    for sent in test_set.belong:
        for aspect in sent:
            if right[aspect] == 0:
                continue
        acc += 1
    f1 = f1_score(true, predicted, average='weighted')
    return acc/len(test_set), f1


Vanilla_GAT = GloVeGAT(embedding_vectors).to(device)

train(Vanilla_GAT, num_epochs)
acc, f1 = test(Vanilla_GAT)
print("Test acc: {}, F1: {}".format(acc, f1))
