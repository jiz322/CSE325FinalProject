from block5 import *

batch = list(test_iterator.batch_sampler)[0]
f = ScoreAssigner.forward(batch[0])
d = torch.softmax(f,dim=-1)
batch_loss = criterion(d,batch[1]-1)
