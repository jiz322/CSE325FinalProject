from random import randint
from sklearn.metrics import confusion_matrix

class NBClassifier:
    def __init__(self, classes, data_file):
        self.classes = classes
        self.class_counts = {}
        for c in classes:
            self.class_counts[c] = 1
        self.counts = {}
        for c in classes:
            self.counts[c] = {}
        self.vocab = set()
        self.data_file = open(data_file, 'r')

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
    
    def update_count(self, classification, data):
        if data in self.counts[classification]:
            self.counts[classification][data] += 1
        else:
            self.counts[classification][data] = 1
    
    def train(self, num_samples = 250, force_even_samples=False):
        training_data = self.sample(num_samples, force_even_samples)
        for d in training_data:
            d_class = d[1]
            self.class_counts[d_class] += 1
            for w in d[0]:
                self.vocab.add(w)
                self.update_count(d_class, w)

    def get_count(self, classification, data):
        if data in self.counts[classification]:
            return self.counts[classification][data] + 1
        else:
            return 1
    
    def test(self, num_samples=25, force_even_samples=False):
        test_data = self.sample(num_samples, force_even_samples)
        tests = []
        for d in test_data:
            real_class = d[1]
            words = set(d[0])
            class_preds = {}
            for c in self.classes:
                pred = self.class_counts[c]
                for w in words:
                    pred *= self.get_count(c, w)
                class_preds[c] = pred
            tests.append((real_class, class_preds))
        return tests

c = NBClassifier([1, 2, 3, 4, 5], 'amazon_total.csv')
num_training_samples = 10000
num_test_samples = 2000
c.train(num_training_samples, True)
classes = c.test(num_test_samples, True)
c.data_file.close()
num_correct = 0
total_error = 0
actual_classes = []
predicted_classes = []
for (c, p) in classes:
    best_class = 1
    best_class_num = 0
    for pred in p:
        if p[pred] > best_class_num:
            best_class_num = p[pred]
            best_class = pred
    if c == best_class:
        num_correct += 1
    else:
        total_error += abs(c - best_class)
    actual_classes.append(c)
    predicted_classes.append(best_class)
print("Accuracy = ", float(num_correct)/num_test_samples)
print("Avg. Error = ", float(total_error)/num_test_samples)
con_mat = confusion_matrix(actual_classes, predicted_classes, labels=[1, 2, 3, 4, 5])
for r in con_mat:
    for c in r:
        print(c,'\t',end='')
    print('\n')