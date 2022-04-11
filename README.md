# CSE325FinalProject

## TODO for RNN 

1. too many words make rNN DOES NOT WORK? Try to reduce words by abondon infrequent words. 
2. Try modify architecture. Output dim 10 and reshape to 10*5, each weight is 1*10 vector. Compute 1*10 dot 10*5 get distribution. (a more complex model)
3. Add baseline: pure regression model
4. Add glove embedding 
5. Better testing: No paddings for con_mat generation
6. Make it a regression task: output_dim = 1 => easier model
7. Softmax/sigmoid: would a linear activation function better in this case?
8. Remove dependent of length in forward function. (also be ware of zero pading)
9. More training examples help?