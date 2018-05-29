import torchtext
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

class NaiveBayes:
    def __init__(self, text):
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.array_like = np.zeros((train_iter.batch_size, len(text.vocab)))
        
    def binarize_occurrences(self, indices):
        occurrences = self.array_like.copy()
        for idx, entry in enumerate(indices): occurrences[idx][entry] = 1
        return occurrences

    def batch_to_input(self, batch, train = True):
        word_indices = batch.text.data.numpy().T
        x = self.binarize_occurrences(word_indices)
        if train:
            y = batch.label.data.numpy()
            return x, y
        else:
            return x

    def train_mnb(self, train_iter, val_iter, no_epochs):
        self.model = MultinomialNB(alpha=1.0, fit_prior=True)
        for epoch in tqdm(range(1, no_epochs+1)):
            for batch in train_iter:
                x, y = self.batch_to_input(batch, train = True)
                self.model.partial_fit(x, y, classes = [1,2])
            
            if epoch % 1 == 0:
                acc = self.validate(val_iter)
                print('Epoch ', epoch, '| Validation Accuracy: ', acc)
        print('Done training.')
        
    def test(self, test_iter):
        "All models should be able to be run with following command."
        upload, trues = [], []

        for batch in test_iter:
            x, y = self.batch_to_input(batch, train = False), batch.label
            probs = self.model.predict(x)
            upload += list(probs)
            trues += list(y.data)
        correct = sum([1 if i == j else 0 for i, j in zip(upload, trues)])
        accuracy = correct / len(trues)
        print('Test Accuracy: ', accuracy)
        
        with open("predictions.txt", "w") as f:
            for u in upload:
                f.write(str(u) + "\n")
                
    def validate(self, val_iter):
        y_p, y_t, correct = [], [], 0
        for batch in val_iter:
            x, y = self.batch_to_input(batch, train = False), batch.label
            probs = self.model.predict(x)[:len(y.data)]
            y_p += list(probs)
            y_t += list(y.data)
        correct = sum([1 if i == j else 0 for i, j in zip(y_p, y_t)])
        accuracy = correct / len(y_p)
        return accuracy

text = torchtext.data.Field(include_lengths = False)
label = torchtext.data.Field(sequential=False)
train, val, test = torchtext.datasets.SST.splits(text, label, filter_pred=lambda ex: ex.label != 'neutral')
text.build_vocab(train)
label.build_vocab(train)
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train, val, test), batch_size=10, device=-1, repeat = False)
mnb = NaiveBayes(text)
mnb.train_mnb(train_iter, val_iter, 1)
mnb.test(test_iter)

