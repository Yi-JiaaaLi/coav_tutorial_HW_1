#%%
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time

def one_hot(x, k, dtype = np.float32):
    return  np.array(x[:,None]==np.arange(k), dtype)

path = os.getcwd()
data = np.load(os.path.join(path, "mnist.npz"))
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]
#normalize
x_train, x_test = x_train/255.0, x_test/255.0
#one-hot encoding
num_labels = 10
y_train,y_test = one_hot(y_train.astype('int32'), num_labels),one_hot(y_test.astype('int32'),num_labels)
#reshape
x_train = x_train.reshape(60000, 28**2)
x_test = x_test.reshape(10000, 28**2)
data.close()
#%%
class Network(): 
    def __init__(self, sizes, activation):
        self.sizes = sizes
        self.activation = activation
        self.weight = None
        self.grads = None
        self.initialization()
            
#%%       
    def initialization(self):
        #number of nodes in each layer
        input =self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output = self.sizes[3]
        self.W1 = np.random.randn(hidden_1, input) *0.1
        self.W2 = np.random.randn(hidden_2, hidden_1) *0.1
        self.W3 = np.random.randn(output, hidden_2) *0.1
        
        self.b1 = np.zeros((hidden_1,1))
        self.b2 = np.zeros((hidden_2,1))
        self.b3 = np.zeros((output,1))

        self.weight = {'W1':self.W1,'b1':self.b1,'W2':self.W2,'b2':self.b2,'W3':self.W3,'b3':self.b3}
  
    def relu(self, x):
        return (x >= 0) * x

    def relu_deriv(self, x):
        return (x >= 0) * 1 

    def softmax(self, x):
        #compute the softamx in a numerically stable way
        x = x - np.max(x)
        sx = np.exp(x) / np.sum(np.exp(x), axis=0)
        return sx
#%%    
    def feedforward(self, x):
        # batch_size, 784
        self.x = x
        self.Z1= np.matmul(self.W1, self.x.T) + self.b1
        self.A1 = self.relu(self.Z1)        
        
        self.Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.A2 = self.relu(self.Z2)      
       
        self.Z3= np.matmul(self.W3, self.A2) + self.b3
        # self.Z3 = np.swapaxes(self.Z3, 0, 1)
        # print("Z3 shape:", self.Z3.shape)
        # Z3.shape(10, 64)
        self.A3= self.softmax(self.Z3)      
       
        return self.A3
#%%  
    def get_loss(self, y_pred, y_gt):
        m = y_gt.shape[0]
        delta = 1e-7
        loss = -np.sum(np.multiply(np.log(y_pred + delta) ,y_gt.T))/ m
        return np.squeeze(loss) 

    # def shuffle(self):
    #     idx = [i for i in range(self.input.shape[0])]
    #     np.random.shuffle(idx)
    #     self.input = self.input[idx]
    #     self.output= self.output[idx]
#%%       
    def backprop(self, y, output):
        m = y.shape[0]
        dZ3 = output - y.T
        dW3 = (1.0/m) * np.matmul(dZ3, self.A2.T)
        db3 = (1.0/m) * np.sum(dZ3, axis = 1, keepdims = True)

        dA2 = np.matmul(self.W3.T, dZ3)
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = (1.0/m) * np.matmul(dZ2, self.A1.T)
        db2 = (1.0/m) * np.sum(dZ2, axis = 1, keepdims = True)
            
        dA1 = np.matmul(self.W2.T, dZ2)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = (1.0/m) * np.matmul(dZ1, self.x)
        db1 = (1.0/m) *np.sum(dZ1, axis = 1, keepdims = True)

        self.grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2,'W3':dW3,'b3':db3}

    def update_weights(self,lr):
        # if self.optimizer == 'sgd':
        for i in self.grads:
            self.weight[i] -= lr * self.grads[i]

            # for key in ('W1','b1','W2','b2','W3','b3'):
            #     self.params[key]= self.params[key] - lr * self.grads[key]
        # else:
            # raise ValueError("Non-supported optimizer.")
    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1)== np.argmax(output.T,axis=-1))

    def train(self,x_train, y_train, x_test, y_test,batch_size,epochs, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        batches_num = -(-x_train.shape[0] // self.batch_size)
        self.optimizer = optimizer
        start_time = time.time()
        template = "epoch {}: {:.2f}s, train_acc={:.2f}, train_loss={:.2f}, test_acc={:.2f}, test_loss={:.2f}"
        train_loss_list=[]
        train_acc_list=[]
        test_acc_list=[]
        
        self.initialization()
        for iteration in range(self.epochs):
            
            # self.shuffle()
            # self.optimizer = optimizer
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]
            for j in range(batches_num):
                #batch
                start = self.batch_size*j
                end = min(start + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[start:end]
                y = y_train_shuffled[start:end]
                output = self.feedforward(x)
                self.backprop(y, output)
                self.update_weights(lr)
            #training data
            output = self.feedforward(x_train)
            train_acc = self.accuracy(y_train, output)
            train_loss = self.get_loss(y_train, output)/60000

            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            #testing data
            output = self.feedforward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.get_loss(y_test, output)/10000

            test_acc_list.append(test_acc)
            print(template.format(iteration+1, time.time()-start_time,train_acc,train_loss,test_acc,test_loss))

        # plot acc
        x=range(len(train_acc_list))
        y1=train_acc_list
        y2=test_acc_list
        plt.figure(figsize=(10,8), dpi=100)
        plt.plot(x,y1,label='train_acc')
        plt.plot(x,y2,linestyle='--',label='test_acc')
        plt.xlabel('epochs')
        plt.ylabel('accurancy')
        plt.legend()
        plt.show()

        #plot loss
        x=range(epochs)
        y=train_loss_list
        plt.figure(figsize=(10,8), dpi=100)
        plt.plot(x,y)
        plt.xlabel('learning time')
        plt.ylabel('value of loss function')
        plt.show()

        #plot 10 images and use model to predict which digit it is
        random_idx = [random.randint(0, 10000) for j in range(0, 10)]
        for i in range(10):
            a = np.reshape(x_test[i], (28, 28))
            plt.subplot(2,5,i+1)
            plt.imshow(a,cmap='gray')
            plt.title(str(np.argmax(y_test[i]))+'→'+str(np.argmax(self.feedforward(np.expand_dims(x_test[i], 0)))))
            plt.axis('off')
        plt.show()
       
fnn = Network(sizes =[784, 100, 150, 10], activation =['relu', 'relu', 'softmax'])
fnn.train(x_train, y_train, x_test, y_test, epochs=50, batch_size=64, optimizer='sgd', lr =1e-2)


