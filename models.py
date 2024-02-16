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

        ker1size = (7, 7)
        ker2size = (3, 3)
        # previos 20, 40, 80
        chans1 = 20
        chans2 = 40
        chans3 = 80
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

class DecoderOnlyTransformer(torch.nn.Module): 
    # decoder only transformer 
    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, d_ffn, mask="causal", 
                 dropout=0.1, device=None, dtype=None, max_seq=5000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim 
        self.num_blocks = num_blocks

        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.posencoder = layers.PositionalEncoder(embed_dim, dropout=dropout, max_seq_len=max_seq)
        decoder_list =[]
        for i in range(num_blocks):
            decoder_list.append(
                layers.TransformerEncoderBlock(embed_dim, num_heads, d_ffn, dropout=dropout, 
                                               device=device, dtype=dtype)
            )
        self.decoders = torch.nn.ModuleList(decoder_list)
        self.final_ffn = layers.Linear(embed_dim, vocab_size)

    def forward(self, input, input_mask="causal", hidden_mask=None): 
        # input: (batch, seq_len, vocab_size)
        encoded = self.posencoder(self.embedding(input))
        out = self.decoders[0](encoded, mask=input_mask)
        for lay in self.decoders[1:]:
            out = lay(out, mask=hidden_mask)
        return functions.softmax(self.final_ffn(out), dim_sum=-1)

    @torch.no_grad
    def get_classification_accuracy(self, X, y, batch_size=64, input_mask="causal", 
                                    hidden_mask=None):
        # input: (batch, seq_len, vocab_size)
        self.eval()
        correct_predict = 0
        n_batches = X.shape[0] // batch_size

        for i in range(n_batches): 
            start = i * batch_size 
            end = min(start+batch_size, X.shape[0])
            X_batch = X[start:end]
            y_batch = y[start:end]
            y_pred = self(X_batch, input_mask=input_mask, hidden_mask=hidden_mask)[:, -1, :]
            correct_predict += torch.sum(torch.argmax(y_batch, dim=-1) 
                                         == torch.argmax(y_pred, dim=-1)).item()
        return correct_predict / y.shape[0]