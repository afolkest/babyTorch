import layers 
import functions 
import torch
from torch import Tensor
import matplotlib.pyplot as plt 

class Optimizer_SGD():
    def __init__(self, learning_rate=1e-2): 
        self.learning_rate = learning_rate 

    def step(self, model): 
        with torch.no_grad(): 
            for param in model.parameters():
                    param -= self.learning_rate * param.grad

class Optimizer_Adam():
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999): 
        self.epsilon = 1e-7
        self.learning_rate = learning_rate 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.t = 0
        self.fresh = True

    def step(self, model): 
        self.t += 1
        with torch.no_grad(): 
            if self.fresh: 
                self.m = [torch.zeros_like(p) for p in model.parameters()]
                self.v = [torch.zeros_like(p) for p in model.parameters()]
                self.fresh = False 

            for i, param in enumerate(model.parameters()):
                    self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * param.grad 
                    self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * param.grad ** 2
                    m_hat = self.m[i] / ( 1-self.beta1 ** self.t )
                    v_hat = self.v[i] / ( 1-self.beta2 ** self.t )
                    param -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)

class Trainer():
    def __init__(self, optimizer, save_path = None):
        self.optimizer = optimizer 
        self.loss_hist = []
        self.train_accuracy_hist = [] 
        self.valid_accuracy_hist = []
        self.cur_best_accuracy = 0
        self.save_path = save_path 

    def train(self, model, X, y, batch_size, epochs, learning_rate=1e-2, saveBest=True, 
              X_val=None, y_val=None, lr_schedule=None):

        model.training = True

        for epoch in range(epochs): 
            running_loss = 0.0

            # Randomize the order of the data
            indices = torch.randperm(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            n_batches = X.shape[0]//batch_size

            for i in range(n_batches): 
                start = i * batch_size 
                end = min(start + batch_size, X.shape[0])
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                pred_batch = model(X_batch)

                loss = functions.cross_entropy(pred_batch, y_batch)
                self.loss_hist.append(loss.item())

                # Zero out gradients before backpropagation
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                loss.backward()

                with torch.no_grad(): 
                    for param in model.parameters():
                            param -= learning_rate * param.grad

                running_loss += loss.item()
                if i % 100 == 0:  
                    print(f'[Epoch: {epoch + 1}, batch: {i + 1}]/{n_batches} \
                        loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            # Evaluate accuracy on test and valid set (if present)
            model.training = False 
            train_acc = model.get_classification_accuracy(X, y, batch_size)
            self.train_accuracy_hist.append(train_acc)
            if X_val is not None:
                valid_acc = model.get_classification_accuracy(X_val, y_val, batch_size)
                self.valid_accuracy_hist.append(valid_acc)
            model.training = True 

            # Save model, if performance is the best so far
            cur_accuracy = train_acc if X_val is None else valid_acc 
            if cur_accuracy > self.cur_best_accuracy:
                self.cur_best_accuracy = cur_accuracy
                if self.save_path is not None:
                    torch.save(model.state_dict(), self.save_path)

            print(f'====================')
            epochstr = f'[Epoch: {epoch + 1}, test accuracy: {train_acc:.3f}' 
            if X_val is not None:
                epochstr =epochstr+ f', validation accuracy: {valid_acc:.3f}'
            print(epochstr)
            print(f'Best accuracy: {self.cur_best_accuracy:.3f}')
            print(f'====================')

    def plot(self):
        fig1, ax1 = plt.subplots()
        ax1.plot(self.loss_hist)
        ax1.set_xlabel('seen batches')
        ax1.set_ylabel('loss')

        fig2, ax2 = plt.subplots()
        ax2.plot(self.train_accuracy_hist, label="train accuracy")
        ax2.plot(self.valid_accuracy_hist, label="validation accuracy")
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')

        plt.show()

