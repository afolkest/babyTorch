import layers 
from layers import Convolution_2d, MaxPool_2d, Linear, Dropout_1d
import functions 
from functions import relu
import torch
from torch import Tensor
import matplotlib.pyplot as plt 


class MLP(torch.nn.Module):
    """
    A Multilayer Perceptron (MLP) class using PyTorch.

    Attributes:
        layers (torch.nn.ModuleList): A list of Linear layers in the MLP.
        layernum (int): The number of layers in the MLP.

    Args:
        layer_sizes (list[int]): List containing the sizes of each layer in the MLP. list[0]
            corresponds to the input size to the first layer of neurons, of which there 
            are layer_sizes[1] in number.
        weight_init_scheme (str, optional): Scheme for initializing weights. 
            Defaults to 'kaiming'. Options include 'kaiming', 'lecun', and 'xavier'.
        device: The device on which to allocate the MLP's parameters.
        dtype: The data type for the MLP's parameters.
    """

    def __init__(self, layer_sizes, weight_init_scheme="kaiming", device=None, dtype=None):
        super().__init__()
        kwargs = {'weight_init_scheme': weight_init_scheme, 'device': device, 'dtype': dtype}

        self.layernum = len(layer_sizes)
        laylist =[]
        for i in range(self.layernum-1):
            laylist.append(Linear(layer_sizes[i], layer_sizes[i+1], **kwargs))
        self.layers = torch.nn.ModuleList(laylist)

    def __str__(self) -> str:
        return "\n".join(list(layer.__str__() for layer in self.layers))

    def forward(self, x_batch: Tensor) -> Tensor:
        output = x_batch
        for i in range(len(self.layers)):
            output = self.layers[i](output)
            if i < len(self.layers)-1:
                output = functions.relu(output)
            else: 
                output = functions.softmax(output, 1)
        return output

    @torch.no_grad
    def get_classification_accuracy(self, X, y, batch_size=64):
        """
        Calculate the classification accuracy of the model, assuming the largest value 
        of self(X) is taken as the final prediction. 

        Args:
            X (Tensor): Input data.
            y (Tensor): True labels.
            batch_size (int, optional): Batch size for processing. Defaults to 64.

        Returns:
            float: The classification accuracy as a percentage.
        """
        correct_predict = 0
        n_batches = X.shape[0] // batch_size

        for i in range(n_batches): 
            start = i * batch_size 
            end = min(start+batch_size, X.shape[0])
            X_batch = X[start:end]
            y_batch = y[start:end]
            y_pred = self(X_batch)
            correct_predict += torch.sum(torch.argmax(y_batch, dim=-1) 
                                         == torch.argmax(y_pred, dim=-1)).item()
        return correct_predict / y.shape[0]


class CNN(torch.nn.Module): 
    def __init__(self,  in_imgsize, in_channels, device=None, dtype=None):
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}

        ker1size = (4, 4)
        ker2size = (2, 2)
        chans1 = 10
        chans2 = 20
        chans3 = 40
        p_dropout = 0.4
        linearsize = 500

        self.conv1 = Convolution_2d(in_channels, chans1, in_imgsize, ker1size, **kwargs)
        self.maxpool1 = MaxPool_2d(self.conv1.out_imgsize, **kwargs)
        self.conv2 = Convolution_2d(self.conv1.out_channels, chans2, self.maxpool1.out_imgsize,  ker2size, **kwargs)
        self.conv3 = Convolution_2d(self.conv2.out_channels, chans2, self.conv2.out_imgsize, ker2size, **kwargs)
        self.maxpool2 = MaxPool_2d(self.conv3.out_imgsize, **kwargs)
        self.conv4 = Convolution_2d(self.conv3.out_channels, chans3, self.maxpool2.out_imgsize, ker2size, **kwargs)
        self.conv5 = Convolution_2d(self.conv4.out_channels, chans3, self.conv4.out_imgsize, ker2size, **kwargs)
        self.maxpool3 = MaxPool_2d(self.conv5.out_imgsize, **kwargs)
        conv_output_size = self.maxpool3.out_imgsize[0]*self.maxpool3.out_imgsize[1] \
                            * self.conv5.out_channels
        self.linear1 = Linear(conv_output_size, linearsize, **kwargs)
        self.linear2 = Linear(linearsize, 10, **kwargs)
        self.dropout = Dropout_1d(p_dropout)

    def forward(self, X): 
        out = relu(self.conv1(X))
        out = self.maxpool1(out)
        out = relu(self.conv2(out))
        out = relu(self.conv3(out))
        out = self.maxpool2(out)
        out = relu(self.conv4(out))
        out = relu(self.conv5(out))
        out = self.maxpool3(out)
        out = self.dropout(torch.flatten(out, start_dim=1, end_dim=3))
        out = self.dropout(self.linear1(out))
        out = self.linear2(out)
        return functions.softmax(out, 1)

    @torch.no_grad
    def get_classification_accuracy(self, X: Tensor, y: Tensor, batch_size=64) -> float:
        self.eval()
        correct_predict = 0
        n_batches = X.shape[0] // batch_size

        for i in range(n_batches): 
            start = i * batch_size 
            end = min(start+batch_size, X.shape[0])
            X_batch = X[start:end]
            y_batch = y[start:end]
            y_pred = self(X_batch)
            correct_predict += torch.sum(torch.argmax(y_batch, dim=-1) 
                                         == torch.argmax(y_pred, dim=-1)).item()
        return correct_predict / y.shape[0]
    
class SGD_Trainer():
    def __init__(self, save_path = None):
        self.loss_hist = []
        self.train_accuracy_hist = [] 
        self.valid_accuracy_hist = []
        self.cur_best_accuracy = 0
        self.save_path = save_path 

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
                if i % 300 == 99:  
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

