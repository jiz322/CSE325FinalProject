from block4 import *
#- _ - _ - _ - _ - _ -


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- The Iterator is a Dataloader object. 
# -- Use for loop in iterator.batch_sampler to access each batches
# -- In this case, each batches is having length 128

# -- Need to Figure out: The way to compute loss for RNN
def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0
    num_batchs = 0
    total_pairs = 0

    # batch[0]: the word batch
    # batch[1]: the tag batch (target)
    for i, batch in enumerate(iterator.batch_sampler):
        num_batchs += 1
        z = ScoreAssigner.forward(batch[0])
        #a = torch.softmax(z,dim=-1)
        loss = 0
        # softmax of logit
        d = torch.softmax(z,dim=-1)
        # cross entropy loss of softmax and score
        loss=criterion(d,batch[1]-1)/BATCH_SIZE
        loss.backward()
        # Clips gradient norm of an iterable of parameters.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss 

def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0
    num_epochs = 0

    for i, batch in enumerate(iterator.batch_sampler):
        num_epochs += 1
        z = ScoreAssigner.forward(batch[0])
        loss = 0
        # softmax of logit
        d = torch.softmax(z,dim=-1)
        # cross entropy loss of softmax and score
        loss=criterion(d,batch[1]-1)/BATCH_SIZE
                
        epoch_loss += loss.item()

    return epoch_loss 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


