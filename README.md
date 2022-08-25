# hw1_feedforward classification network from scratch
feedforward neural network model architecture: fnn =network(sizes=[784,100,150,10])
- 2 hidden layers/mini-batch SGD/cross entrophy loss 
- hyperparameter: epochs=50, batch_size=128, learning_rate=0.01
- batch_size=128

![128_acc](https://user-images.githubusercontent.com/102316495/186373608-bd7ca25f-dbf7-4e15-a15c-8f74a434450f.png "accuracy")
![128_loss](https://user-images.githubusercontent.com/102316495/186373939-2abf3c0c-1e77-4631-9b2d-1ea269828e7e.png "loss")
![128_predict](https://user-images.githubusercontent.com/102316495/186376213-85c14c35-087c-46e1-b4fe-9f04d1e2c191.png "predict digit")

- Repeat 1 with different batch sizes: change batch_size to 32

![32_acc](https://user-images.githubusercontent.com/102316495/186375128-31281fad-43de-4c60-88ef-c8aff50ed50b.png "acc")
![32_loss](https://user-images.githubusercontent.com/102316495/186375142-3441b24a-f795-41d1-b40c-550c8ab6df35.png "loss")
- initialize weights to zero: if weights of neural network are initialized with 0,the derivative with respect to loss function is the same for every W,thus all weights have the same value in subsequent iterations.this makes hidden neurons symmetric.

