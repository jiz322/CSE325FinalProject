from block3 import *

# _-_-_-_-_-_-_-_-_-_-
from torch import optim
import time
import math

BATCH_SIZE = 128


corpora = Corpora()

corpora.read_corpus(True)
corpora.read_corpus(False)

print(f'Number of training sentences = {len(corpora.training_reviews)}')
print(f'Number of test sentences = {len(corpora.test_reviews)}')
print(f'Number of unique input tokens = {len(corpora.word_index)}')
print(f'Maximal sentence length = {corpora.max_len}')

print("\n\n Creating training Dataset, Sampler, and Iterators...")
training_dataset = ReviewRateDataset(corpora.training_reviews)
training_sampler = SortedBatchSampler(training_dataset, batch_size=BATCH_SIZE)
training_iterator = DataLoader(training_dataset,
                                  collate_fn = padding_collate_func,
                                  batch_sampler = training_sampler)
print("\n\n Creating test Dataset, Sampler, and Iterators")
test_dataset = ReviewRateDataset(corpora.test_reviews)
test_sampler = SortedBatchSampler(test_dataset, batch_size=BATCH_SIZE)
test_iterator = DataLoader(test_dataset,
                              collate_fn = padding_collate_func,
                              batch_sampler = test_sampler)


INPUT_DIM = len(corpora.word_index)
OUTPUT_DIM = 5
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 1 # number of LSTM layers.
BIDIRECT = 0 # 0: single direction (the default setting); 1: bidirectional
DROPOUT = 0.5
# initialize the model
ScoreAssigner = LSTMScoreAssigner(INPUT_DIM, OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, BIDIRECT)#.cuda(3)



# Glove Embedding here?
def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

ScoreAssigner.apply(init_weights)

optimizer = optim.Adam(ScoreAssigner.parameters())

# we use 0 to represent padded POS tags and the loss function should ignore that.
# we calculate the sum of losses of pairs in each batch
PAD_INDEX = 0


# input: vector of [length, output_dim], integer (score)
criterion = nn.CrossEntropyLoss(reduction = 'sum', ignore_index = PAD_INDEX)



N_EPOCHS = 10
CLIP = 1

best_test_loss = float('inf')

training_losses = []
test_losses = []

# -- After comment all of these out
# -- I can safely import the how thing in python consoler