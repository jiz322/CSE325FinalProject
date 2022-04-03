from block3 import *
from test_b2 import *
INPUT_DIM = len(a.word_index.keys())
OUTPUT_DIM = 5
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 1 # number of LSTM layers.
BIDIRECT = 0 # 0: single direction (the default setting); 1: bidirectional
DROPOUT = 0.5

sa = LSTMScoreAssigner(INPUT_DIM, OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, BIDIRECT)
# e = sa.embedding(list(test_sampler)[0][0])
# output = sa.rnn(e)[0]
# logit=sa.fc(output)

# SHAPE: [batchsize, sentence_length, output_dim]
# [128,419,5]
# 128: batchsize
# 419: length
# 5: 5 scores total
f = sa.forward(list(test_sampler)[0][0])
