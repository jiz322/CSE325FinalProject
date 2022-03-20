from sklearn import svm
from random import randint
from sklearn.metrics import confusion_matrix

class SVM:
    def __init__(self, classes, data_file):
        self.classes = classes
        self.data_file = open(data_file, 'r')
        self.vocab = {}
        self.model = svm.SVC(decision_function_shape='ovr', kernel='sigmoid')

    def sample(self, num_samples, force_even_samples=False):
        samples = [''] * num_samples
        if force_even_samples:
            num_classes = {}
            for c in self.classes:
                num_classes[c] = 0
            max_class_samples = num_samples // len(self.classes) + 1
        fh = self.data_file
        offset = randint(0, 1000)
        for i in range(offset):
            fh.readline()
        for i in range(num_samples):
            discard_lines = randint(10, 20)
            for j in range(discard_lines):
                fh.readline()
            line = fh.readline()[:-1]
            line = line.split(',')
            line[1] = int(line[1])
            if force_even_samples:
                while num_classes[line[1]] >= max_class_samples:
                    line = fh.readline()[:-1]
                    line = line.split(',')
                    line[1] = int(line[1])
                num_classes[line[1]] += 1
            line[0] = line[0].split(' ')
            line[0] = [l for l in line[0] if l != '.' and l != '!']
            samples[i] = line
        return samples
    
    def add_to_vocab(self, words):
        for w in words:
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)

    def get_word_indices(self, words):
        indices = [0] * len(self.vocab)
        for word in self.vocab:
            indices[self.vocab[word]] = words.count(word)
        return indices
    
    def train(self, num_samples, force_even_sampling=False):
        samples = self.sample(num_samples, force_even_sampling)
        [self.add_to_vocab(s[0]) for s in samples]
        X = []
        Y = []
        for s in samples:
            x = self.get_word_indices(s[0])
            X.append(x)
            Y.append(s[1])
        self.model.fit(X, Y)
    
    def test(self, num_samples, force_even_sampling=False):
        samples = self.sample(num_samples, force_even_sampling)
        samples.sort(key=lambda s: s[1])
        X = []
        Y_actual = []
        for s in samples:
            x = self.get_word_indices(s[0])
            X.append(x)
            Y_actual.append(s[1])
        results = self.model.decision_function(X)
        Y_pred = []
        for r in results:
            r = list(r)
            Y_pred.append(r.index(max(r)) + 1)
        return (Y_actual, Y_pred)


my_svm = SVM([1, 2, 3, 4, 5], 'amazon_total.csv')
my_svm.train(1000, True)
(actual, pred) = my_svm.test(200, True)
my_svm.data_file.close()
num_correct = 0
tot_error = 0
for i in range(len(actual)):
    if actual[i] == pred[i]:
        num_correct += 1
    else:
        tot_error += abs(actual[i] - pred[i])
print("Accuracy = ", float(num_correct)/len(actual))
print("Avg. Error = ", float(tot_error)/len(actual))
con_mat = confusion_matrix(actual, pred, labels=[1, 2, 3, 4, 5])
for r in con_mat:
    for c in r:
        print(c,'\t',end='')
    print('\n')