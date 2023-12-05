from Data import *
from Model import *
import torch
import numpy as np
import time
from transformers import get_linear_schedule_with_warmup, AdamW


aux_extract = 'What are the aspect terms in the sentence?'
aux_polarity = 'What is the sentimental polarity of {} in the sentence?'

sent_map = {
    'positive': 0,
    'negative': 1,
    'conflict': 2,
    'neutral': 3
}

num_labels = 4
num_epochs = 5
total_steps = len(train_sent)
learning_rate = 2e-5
span_threshold = 0.5
extract_weight = 1
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model):
    model.eval()
    total_loss = 0
    for item in val_sent:
        aspects = [" ".join(a['term']) for a in item['aspects']]
        sentiments = [torch.LongTensor([sent_map[a['polarity']]]) for a in item['aspects']]
        loss = model.learn(" ".join(item['token']), aspects, sentiments, aux_extract, aux_polarity, tokenizer, criterion, extract_weight, span_threshold)
        total_loss += loss.item()
    return total_loss / len(val_sent)


def train(model, epochs):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=epochs*total_steps)
    for i in range(epochs):
        print('========= Epoch {} / {} =========='.format(i+1, epochs))
        t0 = time.time()
        total_train_loss = 0
        for step, item in enumerate(train_sent):
            if step % 40 == 0 and not step == 0:
                elapsed = round(time.time() - t0, 2)
                print('Batch {} of {}. Elapsed: {}.'.format(step, len(train_sent), elapsed))
            aspects = [" ".join(a['term']) for a in item['aspects']]
            sentiments = [torch.LongTensor([sent_map[a['polarity']]]) for a in item['aspects']]
            optimizer.zero_grad()
            loss = model.learn(" ".join(item['token']), aspects, sentiments, aux_extract, aux_polarity, tokenizer, criterion, extract_weight, span_threshold)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_sent)
        training_time = round(time.time()-t0, 2)
        print("Average training loss: {}".format(avg_train_loss))
        print("Training epoch took: {}".format(training_time))
        val_loss = evaluate(model)
        print("Val loss: {}".format(val_loss))
    print("Training complete!")


def test(model):
    model.eval()
    acc = 0
    for item in test_sent:
        aspects = [" ".join(a['term']) for a in item['aspects']]
        sentiments = [torch.LongTensor([sent_map[a['polarity']]]) for a in item['aspects']]
        acc += model.predict(" ".join(item['token']), aspects, sentiments, aux_polarity, tokenizer)
    return acc/len(test_sent)


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dual_mrc = DualMRC()

train(dual_mrc, num_epochs)
print("Test acc:", test(dual_mrc))
