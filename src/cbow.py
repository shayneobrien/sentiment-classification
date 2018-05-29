import torchtext
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchtext.vocab import Vectors
from tqdm import tqdm

class CBoW(nn.Module):
    def __init__(self, input_size, num_classes, batch_size):
        super(CBoW, self).__init__()
        self.embeddings = nn.Embedding(text.vocab.vectors.size()[0], text.vocab.vectors.size()[1])
        self.embeddings.weight.data.copy_(text.vocab.vectors)
        self.linear = nn.Linear(input_size+1, num_classes, bias = True)
    
    def forward(self, x):
        x, lengths = x
        lengths = Variable(lengths.view(-1, 1).float())
        embedded = self.embeddings(x)
        average_embed = embedded.mean(0)
        concat = torch.cat([average_embed, lengths], dim = 1) # add lengths as a feature
        output = self.linear(concat)
        logits = nn.functional.log_softmax(output, dim = 1)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        return logits.max(1)[1] + 1
    
    def train(self, train_iter, val_iter, test_iter, num_epochs, learning_rate = 1e-3, plot = False):
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        loss_vec = []
        
        for epoch in tqdm(range(1, num_epochs + 1)):
            epoch_loss = 0
            for batch in train_iter:
                x = batch.text
                y = batch.label
                
                optimizer.zero_grad()
                
                y_p = self.forward(x)
                
                loss = criterion(y_p, y-1)
                loss.backward()
                
                optimizer.step()
                epoch_loss += loss.data[0]
                
            self.model = model
            
            loss_vec.append(epoch_loss / len(train_iter))
            if epoch % 1 == 0:
                acc = self.validate(val_iter)
                print('Epoch {} loss: {} | acc: {}'.format(epoch, loss_vec[epoch-1], acc))
                self.model = model
                self.test(test_iter)
        
        if plot:
            plt.plot(range(len(loss_vec)), loss_vec)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        print('\nModel trained.\n')
        self.loss_vec = loss_vec
        self.model = model

    def test(self, test_iter):
        "All models should be able to be run with following command."
        upload, trues = [], []
        # Update: for kaggle the bucket iterator needs to have batch_size 10
        for batch in test_iter:
            # Your prediction data here (don't cheat!)
            x, y = batch.text, batch.label
            preds = self.predict(x)
            upload += list(preds.data.numpy())
            trues += list(y.data.numpy())
            
        correct = sum([1 if i == j else 0 for i, j in zip(upload, trues)])
        accuracy = float(correct) / len(trues)
        print('Test Accuracy:', accuracy)

        with open("predictions.txt", "w") as f:
            for u in upload:
                f.write(str(u) + "\n")
                
    def validate(self, val_iter):
        y_p, y_t, correct = [], [], 0
        for batch in val_iter:
            x, y = batch.text, batch.label
            probs = self.model.predict(x)[:len(y.data.numpy())]
            y_p += list(probs.data.numpy())
            y_t += list(y.data.numpy())
        correct = sum([1 if i == j else 0 for i, j in zip(y_p, y_t)])
        accuracy = float(correct) / len(y_p)
        return accuracy

text = torchtext.data.Field(include_lengths = True)
label = torchtext.data.Field(sequential=False)
train, val, test = torchtext.datasets.SST.splits(text, label, filter_pred=lambda ex: ex.label != 'neutral')
text.build_vocab(train)
label.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train, val, test), batch_size=10, device=-1, repeat = False)
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
text.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

model = CBoW(input_size = 300, num_classes = 2, batch_size = 10)
model.train(train_iter = train_iter, val_iter = val_iter, test_iter = test_iter, num_epochs = 25, learning_rate = 1e-4, plot = False)
model.test(test_iter)

