from block2 import *

a = Corpora()
a.read_corpus(True)
a.read_corpus(False)
print("numbers of test reviews", len(a.test_reviews))
print("numbers of train reviews", len(a.training_reviews))
print("numbers of Unique words", len(a.word_index.keys()))
print(f'Maximal sentence length = {a.max_len}')
test_dataset = ReviewRateDataset(a.test_reviews)
test_sampler = SortedBatchSampler(test_dataset, batch_size=128)
print("lenth is test_sampler is: ", len(test_sampler))

# a batch sampler
try2 = list(test_sampler)[2]
try1 = list(test_sampler)[1]

