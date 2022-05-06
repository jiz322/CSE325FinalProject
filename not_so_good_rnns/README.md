# These are some models that are not so good

The rnn_5_class_emb is actually not so bad, but it does not make much accuracy progress than the version without using rnn. (It gets lower test loss, though) \
The last_hidden_unbalanced is not successfully trained, since it is predicting all reviews with 5 stars right now since there are a lot more 5 stars in the training set. \
The more_linear_padding does not reach accuracy as good as that of having less linear layers. Reason is unsure. \
The unbalanced_test is having bug. It does not assign same index as the trained model it loads.