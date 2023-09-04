import torch
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import MessagePassing, GATConv, RGATConv


class GloVeGAT(torch.nn.Module):
    def __init__(self, embedding_vectors, hidden_channel=100, class_num=3, heads=8, lstm_layers=8, dropout=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_vectors.shape[0], embedding_vectors.shape[1], _weight=embedding_vectors)
        self.lstm_init = torch.nn.Linear(embedding_vectors.shape[1], hidden_channel * heads)
        self.gat1 = GATConv(hidden_channel, hidden_channel, heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channel, hidden_channel, heads, dropout=dropout)
        self.gat3 = GATConv(hidden_channel, hidden_channel, heads, dropout=dropout)
        self.lstm = torch.nn.LSTM(input_size=hidden_channel * heads, hidden_size=hidden_channel, num_layers=lstm_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channel, class_num)

    def forward(self, data, query_indices):
        x, edge_index = data.x, data.edge_index
        x = self.embedding(x)
        out, (h, c) = self.lstm(F.relu(self.lstm_init(x)))
        out, (h, c) = self.lstm(F.relu(self.gat1(out, edge_index)), (h, c))
        out, (h, c) = self.lstm(F.relu(self.gat2(out, edge_index)), (h, c))
        out, _ = self.lstm(F.relu(self.gat3(out, edge_index)), (h, c))
        x = self.linear(out[query_indices].mean(dim=0))
        return x


class GloVeRGAT(torch.nn.Module):
    def __init__(self, embedding_vectors, num_relation, hidden_channel=100, class_num=3, heads=8, lstm_layers=8, dropout=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_vectors.shape[0], embedding_vectors.shape[1], _weight=embedding_vectors)
        self.lstm_init = torch.nn.Linear(embedding_vectors.shape[1], hidden_channel * heads)
        self.gat1 = RGATConv(hidden_channel, hidden_channel, heads, num_relations=num_relation, dropout=dropout)
        self.gat2 = RGATConv(hidden_channel, hidden_channel, heads, num_relations=num_relation, dropout=dropout)
        self.gat3 = RGATConv(hidden_channel, hidden_channel, heads, num_relations=num_relation, dropout=dropout)
        self.lstm = torch.nn.LSTM(input_size=hidden_channel * heads, hidden_size=hidden_channel, num_layers=lstm_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channel, class_num)

    def forward(self, data, query_indices):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.embedding(x)
        out, (h, c) = self.lstm(F.relu(self.lstm_init(x)))
        out, (h, c) = self.lstm(F.relu(self.gat1(out, edge_index, edge_type)), (h, c))
        out, (h, c) = self.lstm(F.relu(self.gat2(out, edge_index, edge_type)), (h, c))
        out, _ = self.lstm(F.relu(self.gat3(out, edge_index, edge_type)), (h, c))
        x = self.linear(out[query_indices].mean(dim=0))
        return x


class RGAT(torch.nn.Module):
    def __init__(self, input_channel=300, hidden_channel=100, class_num=3, heads=8, dropout=0.5, num_relations=50):
        super().__init__()
        self.rgat1 = RGATConv(input_channel, hidden_channel, num_relations, heads, dropout=dropout)
        self.rgat2 = RGATConv(hidden_channel, hidden_channel, num_relations, heads, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_channel, class_num)

    def forward(self, data, query_indices):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = F.relu(self.rgat1(x, edge_index, edge_type))
        x = F.relu(self.rgat2(x, edge_index, edge_type))
        x = self.linear(x[query_indices].mean(dim=0))
        return x


class BertRGAT(torch.nn.Module):
    def __init__(self, hidden_channel=100, class_num=3, heads=8, dropout=0.5, num_relations=50):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.rgat = RGAT(768, hidden_channel, class_num, heads, dropout, num_relations)

    def forward(self, token, data, query_indices):
        data.x = self.bert(**token).last_hidden_state
        x = self.rgat(data, query_indices)
        return x
