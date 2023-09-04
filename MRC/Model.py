import torch
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DualMRC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.start_detect = torch.nn.Linear(768, 2, bias=False)
        self.end_detect = torch.nn.Linear(768, 2, bias=False)
        self.classifier = torch.nn.Linear(768, 4)

    def left(self, token):
        token = token.to(device)
        encoding = self.bert(**token).last_hidden_state
        start_logits = self.start_detect(encoding)
        end_logits = self.end_detect(encoding)
        return start_logits, end_logits

    def right(self, token):
        token = token.to(device)
        encoding = self.bert(**token).last_hidden_state[:, 0]
        logit = self.classifier(encoding)
        return logit

    def multi_span(self, start_logits, end_logits, threshold, m_candidates, max_targets):
        R = []
        U = []
        O = []
        S = torch.sort(start_logits, descending=True)[1][:m_candidates]
        E = torch.sort(end_logits, descending=True)[1][:m_candidates]
        for s in S:
            for e in E:
                if s <= e and start_logits[s] + end_logits[e] >= threshold:
                    u = start_logits[s] + end_logits[e] - (e - s + 1)
                    r = (s, e)
                    R.append(r)
                    U.append(u)
        while R and len(O) < max_targets:
            l = U.index(max(U))
            opt = R[l]
            O.append(opt)
            R.pop(l)
            U.pop(l)
            i = 0
            while i < len(R):
                r = R[i]
                if opt[0] <= r[0] <= opt[1] or opt[0] <= r[1] <= opt[1]:
                    R.pop(i)
                    U.pop(i)
                    i -= 1
                i += 1
        return O

    def learn(self, sent, aspects, sentiments, left_aux, right_aux, tokenizer, criterion, extract_weight, threshold):
        left_token = tokenizer(sent + '[SEP]' + left_aux, return_tensors='pt')
        ids = left_token['input_ids'][0]
        start_label = torch.LongTensor([-100 if wid is None else 0 for wid in left_token.word_ids()])
        end_label = start_label + 0
        for aspect in aspects:  # Assign the token label
            at = tokenizer(aspect, return_tensors='pt')['input_ids'][0][1:-1]
            for i in range(len(ids) - len(at) + 1):
                if torch.equal(ids[i:i + len(at)], at):
                    start_label[i] = 1
                    end_label[i + len(at) - 1] = 1
        start_logits, end_logits = self.left(left_token)
        EALoss = criterion(start_logits.squeeze_(0), start_label.to(device)) + criterion(end_logits.squeeze_(0),
                                                                                         end_label.to(device))
        start_logits = start_logits.cpu()[:, 1]
        end_logits = end_logits.cpu()[:, 1]
        for i, w in enumerate(left_token.word_ids()):
            if w is None:
                start_logits[i] = -100
                end_logits[i] = -100
        i = -1
        ASLoss = 0
        for i, (s, e) in enumerate(self.multi_span(start_logits, end_logits, threshold, 4, len(sentiments))):
            predicted_aspect = tokenizer.decode(ids[s:e + 1])
            sample = tokenizer(sent + '[SEP]' + right_aux.format(predicted_aspect), return_tensors='pt')
            aspect_logits = self.right(sample)
            ASLoss += criterion(aspect_logits, sentiments[i])
        if i < len(sentiments) - 1:
            extract_weight *= 2
        return extract_weight * EALoss + ASLoss

    def predict(self, sent, aspects, sentiments, aux, tokenizer):
        for i, a in enumerate(aspects):
            sample = tokenizer(sent + '[SEP]' + aux.format(a), return_tensors='pt')
            logits = self.right(sample)
            if not logits.argmax().item() == sentiments[i].item():
                return 0
        return 1
