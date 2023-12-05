import torch
from torchtext import vocab
import json
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset, random_split

class_map = {
    'positive': 0,
    'negative': 1,
    'neutral': 2
}

deprel_map = {}

glove = vocab.GloVe()
embedding_vectors = [glove['unk']]
word_map = {'unk': 0}
word_list = ['unk']

with open("./laptop_test.json") as f:
    test = json.load(f)

with open("./laptop_train.json") as f:
    train = json.load(f)

for sent in test + train:
    for i in sent['token']:
        word = i.lower()
        if word in glove.stoi:
            if word not in word_map:
                word_map[word] = len(word_map)
                word_list.append(word)
                embedding_vectors.append(glove[word])
    for i in sent['deprel']:
        if i not in deprel_map:
            deprel_map[i] = len(deprel_map)


embedding_vectors = torch.stack(embedding_vectors)


def word2idx(w):
    w = w.lower()
    if w in word_map:
        return word_map[w]
    return 0


# def reconstruct(heads, aspects):


def head2edge(heads, deprels):
    edge_from = []
    edge_to = []
    edge_type = []
    for n, (head, deprel) in enumerate(zip(heads, deprels)):
        head = int(head)
        if not head == 0:
            edge_from.append(n)
            edge_to.append(head - 1)
            edge_type.append(deprel_map[deprel])
    return torch.tensor([edge_from, edge_to]), torch.tensor(edge_type)


class GloVeGATSet(Dataset):
    def __init__(self, sent_list):
        self.belong = [[] for _ in range(len(sent_list))]
        self.item_list = []
        for n, item in enumerate(sent_list):
            index_list = [word2idx(w) for w in item['token']]
            for aspect in item['aspects']:
                self.belong[n].append(len(self.item_list))
                edge_index, edge_type = head2edge(item['head'], item['deprel'])
                self.item_list.append((Data(x=torch.tensor(index_list), edge_index=edge_index, edge_type=edge_type), aspect['from'], aspect['to'], torch.tensor(class_map[aspect['polarity']])))

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        return self.item_list[index]


domain = GloVeGATSet(train)
train_set, val_set = random_split(domain, [int(0.8*len(domain)), len(domain)-int(0.8*len(domain))])
test_set = GloVeGATSet(test)

