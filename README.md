# Regular-model-compression

cnn compression for keras

English is not my native language,too; please excuse typing errors.

This code is based on Model-Compression-Keras["deep compression"](https://arxiv.org/abs/1510.00149)


### How to use

1. Train the model (normal-cnn)

For CIFAR-10 and compression rate=0.8:

    python train_cnn.py --model=vgg --data=c10

For CIFAR-100 and compression rate=0.8:

    python train_cnn.py --model=vgg --data=c100
    
2. Decode and evaluation:

        python decode_and_evaluate_cnn.py --model=vgg
    
3. Training models with different compression rates

        python regular_compression.py
 
4. View weight distribution
    
         python all_w_distribution.py
         ![image](https://user-images.githubusercontent.com/42563899/114002515-84c23f00-988f-11eb-8157-95b0a5d8acec.png)

### Results

#### CIFAR-10
![image](https://user-images.githubusercontent.com/42563899/114000261-69eecb00-988d-11eb-9e48-e282b0ab16d5.png)

### conclusion

Both L1 and L2 regular terms can promote the weight of the model to converge to 0, and the effect of L1 is more obvious. In addition, it has a good effect on improving the compression efficiency of high pruning rate model. L1 tends to produce a small number of features, while the other features are all zero, because the optimal parameter value is very likely to appear on the coordinate axis, which will lead to a one-dimensional weight of zero, resulting in a sparse weight matrix. L2 will select more features, which will be close to zero. The optimal parameter value is very small, so the parameter of each dimension will not be zero. When ‖w‖ is minimized, each term will be close to zero.
