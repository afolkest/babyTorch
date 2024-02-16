import torch
import math
import functions
from functions import find_same_padding
from torch import Tensor
import numpy as np

class Linear(torch.nn.Module):
    """
    Custom linear layer module for neural networks using PyTorch.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        weights (torch.nn.Parameter): Weights of the linear layer.
        biases (torch.nn.Parameter): Biases of the linear layer.
        n_params (int): Total number of parameters in the layer.

    Args:
        in_features (int): Number of features of the input.
        out_features (int): Number of features of the output.
        weight_init_scheme (str, optional): Scheme to initialize weights.
            Defaults to 'kaiming'. Other options are 'lecun' and 'xavier'.
        device: The device on which to allocate the layer's parameters.
        dtype: The data type for the layer's parameters.
    """

    def __init__(self, in_features: int, out_features: int, weight_init_scheme: str="kaiming", 
                 device=None, dtype=None) -> None:
        """Initialize the MLP with the specified layer sizez and weight init method"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 

        if weight_init_scheme == "kaiming":
            variance = 2 / in_features
        elif weight_init_scheme == "lecun":
            variance = 1 / in_features
        else:
            variance = 2 / (in_features+out_features)
            if weight_init_scheme != "xavier":
                print("warning: unknown weight init scheme, defaults to xavier init") 
        weight_bound = math.sqrt(3 * variance)
        self.weights = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
            .uniform_(-weight_bound, weight_bound)
        )
        self.biases = torch.nn.Parameter(torch.zeros((out_features,), dtype=dtype, device=device))
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, input : Tensor ) -> Tensor:
        return input @ self.weights.T + self.biases
    
    def __str__(self) -> str:
        return "Linear layer with input->output = " \
              + str(self.in_features) + "->" + str(self.out_features) \
              + ". Total params = " + str(self.n_params)

class ConvolutionBase(torch.nn.Module): 
    """
    Base class for convolutional layers.

    Args:
        in_imgsize (tuple): Input image size (height, width).
        kernel (tuple): Convolutional kernel size (height, width).
        stride (tuple): Stride of the convolution operation (height, width).
        padding_type (str): Type of padding to use.
        device (torch.device, optional): Device for the layer parameters and operations.
        dtype (torch.dtype, optional): Data type for the layer parameters and operations. 
    """
    def __init__(self, in_imgsize, kernel, stride=(1,1), padding_type="same", device=None, 
                 dtype=None):
        super().__init__()
        self.kernel = kernel 
        self.stride = stride 
        self.in_imgsize = in_imgsize 
        self.padding_type = padding_type 

        # Compute padding quantites
        if self.padding_type=='same':
            self.height_pad = find_same_padding(self.in_imgsize[0], self.kernel[0], self.stride[0])
            self.width_pad = find_same_padding(self.in_imgsize[1], self.kernel[1], self.stride[1])
            self.out_imgsize = (math.ceil(self.in_imgsize[0] / self.stride[0]), 
                                math.ceil(self.in_imgsize[1] / self.stride[1]))
        else:
            self.height_pad = 0
            self.width_pad = 0 
            self.out_imgsize = (self.in_imgsize[0] // self.stride[0],
                                self.in_imgsize[1] // self.stride[1])

        # Define projector onto space matching the convolution window size 
        self.projector = torch.zeros((self.kernel[0], self.kernel[1], self.out_imgsize[0], 
                                      self.out_imgsize[1], self.in_imgsize[0]+self.height_pad, 
                                      self.in_imgsize[1]+self.width_pad))
        # Extract index meshgrids 
        kh_grid, kw_grid, h_out_grid, w_out_grid = np.ogrid[:self.kernel[0], :self.kernel[1], 
                                                   :self.out_imgsize[0], :self.out_imgsize[1]]
        # See appendix 2 in example2 notebook for this logic
        self.projector[kh_grid, kw_grid, h_out_grid, w_out_grid, 
                       h_out_grid*self.stride[0] + kh_grid,  w_out_grid*self.stride[1]+kw_grid] = 1 

class Convolution_2d(ConvolutionBase):
    """
    2D convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        in_imgsize (tuple): Input image size (height, width).
        kernel (tuple): Convolutional kernel size (height, width).
        stride (tuple): Stride of the convolution operation (height, width). Default is (1, 1).
        padding_type (str): Type of padding to use. Default is "same".
        device (torch.device, optional): Device for the layer parameters and operations. 
        dtype (torch.dtype, optional): Data type for the layer parameters and operations.
    """
    def __init__(self, in_channels, out_channels, in_imgsize, kernel, stride=(1,1), 
                 padding_type='same', device=None, dtype=None):
        super().__init__(in_imgsize, kernel, stride, padding_type, device, dtype)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Kaiming/he normalization of the weights of conv kernel 
        variance = 2 / (kernel[0]*kernel[1]*in_channels)
        weight_bound = math.sqrt(3 * variance)
        self.weights = torch.nn.Parameter(
                        torch.empty((out_channels, in_channels, kernel[0], kernel[1]), dtype=dtype, 
                        device=device).uniform_(-weight_bound, weight_bound)
        )
        self.biases = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, X):
        """X of shapj (batches, feature_maps, height, width)"""
        pad_tuple = (math.floor(self.height_pad/2),
                     math.ceil(self.height_pad/2), 
                     math.floor(self.width_pad/2),
                     math.ceil(self.width_pad/2))
        X_pad = torch.nn.functional.pad(X, pad_tuple)

        #h, w = indices in projected space matching the kernel size
        #i, j = indices in image output space
        #m, n = indices in image input space
        #F, G = indices in feature_map input and output space, respectively
        #B = index in batch space
        X_proj = torch.einsum("hwijmn, BFmn -> BFhwij", self.projector, X_pad)
        return torch.einsum("GFhw, BFhwij -> BGij", self.weights, X_proj) \
                + self.biases.view(1, -1, 1, 1)

    def __str__(self):
        return (f"Convolution_2D(in_channels={self.weights.shape[1]}, out_channels={self.weights.shape[0]}, "
                f"kernel={self.kernel}, stride={self.stride}, padding_type='{self.padding_type}', "
                f"input_size=({self.in_imgsize[0]}, {self.in_imgsize[1]}), "
                f"output_size=({self.out_imgsize[0]}, {self.out_imgsize[1]}), "
                f"padding=(height: {self.height_pad}, width: {self.width_pad}))")

class MaxPool_2d(ConvolutionBase):
    """
    2D max pooling layer.

    Args:
        in_imgsize (tuple): Input image size (height, width).
        kernel (tuple): Max pooling kernel size (height, width). Default is (2, 2).
        stride (tuple): Stride of the max pooling operation (height, width). Default is (2, 2).
        padding_type (str): Type of padding to use. Default is "same".
        device (torch.device, optional): Device for the layer parameters and operations.
        dtype (torch.dtype, optional): Data type for the layer parameters and operations. 
    """
    def __init__(self, in_imgsize, kernel=(2,2), stride=(2,2), padding_type='same', 
                 device=None, dtype=None):
        super().__init__(in_imgsize, kernel, stride, padding_type, device, dtype)

    def forward(self, X):
        pad_tuple = (math.floor(self.height_pad/2),
                     math.ceil(self.height_pad/2), 
                     math.floor(self.width_pad/2),
                     math.ceil(self.width_pad/2))
        X_pad = torch.nn.functional.pad(X, pad_tuple)
        X_proj = torch.einsum("hwijmn, BFmn -> BFhwij", self.projector, X_pad)
        max = torch.max(X_proj, dim=2, keepdim=True).values
        max = torch.max(max, dim=3, keepdim=True).values
        return max.squeeze(dim=(2, 3)) 

    def __str__(self):
        return (f"Maxppol_2D(kernel={self.kernel}, stride={self.stride}, padding_type='{self.padding_type}', "
                f"input_size=({self.in_imgsize[0]}, {self.in_imgsize[1]}), "
                f"output_size=({self.out_imgsize[0]}, {self.out_imgsize[1]}), "
                f"padding=(height: {self.height_pad}, width: {self.width_pad}))")

class Dropout_1d(torch.nn.Module):
    """
    1D dropout layer.

    Args:
        p (float): Dropout probability.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p 

    def forward(self, X):
        if self.p is None:
            return X
        if self.training: 
            mask = torch.bernoulli(torch.full(X.shape, 1-self.p)) / (1-self.p)
            return mask * X
        return X 

class LSTM(torch.nn.Module):
    """
    Long Short-Term Memory (LSTM) layer.

    Args:
        in_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        num_layers (int): Number of LSTM layers. Default is 1.
        p_dropout (float): Dropout probability. Default is 0.
        dtype (torch.dtype, optional): Data type for the layer parameters and operations. 
        device (torch.device, optional): Device for the layer parameters and operations. 
    """
    def __init__(self, in_dim, hidden_dim, num_layers=1, dtype=None, device=None):
        super().__init__()
        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        weight_range =  math.sqrt(1/hidden_dim)
        self.w_hidden_list = torch.nn.ParameterList()
        self.w_in_list = torch.nn.ParameterList()
        self.bias_list = torch.nn.ParameterList()

        for i in range(num_layers):
            # neurons acting on the hidden states 
            self.w_hidden_list.append(torch.nn.Parameter(
                torch.empty((4 * hidden_dim, hidden_dim), dtype=dtype, device=device)
                .uniform_(-weight_range, weight_range))
                )
            # neurons acting on the input. for i=0 this is of size in_dim, while for the higher  
            # layers, the input is the previous hidden state, so this now equals hidden_dim
            input_dim = in_dim if i==0 else hidden_dim 
            self.w_in_list.append(torch.nn.Parameter(
                torch.empty((4 * hidden_dim, input_dim), dtype=dtype, device=device)
                .uniform_(-weight_range, weight_range))
            )
            self.bias_list.append(torch.nn.Parameter(torch.zeros(4 * hidden_dim)))

    def forward(self, X_in, h=None, c=None):
        # X_in: (seq_length, batches, in_dim)
        # h_prev: (layers, batches, hidden_dim)
        # c_prev: (layers, batches, hidden_dim)
        # output: (seq_length, batches, hidden_dim)

        if h is None:
            h = [torch.zeros(X_in.shape[1], self.hidden_dim) for _ in range(self.num_layers)]
        if c is None:
            c = [torch.zeros(X_in.shape[1], self.hidden_dim) for _ in range(self.num_layers)]
        output = torch.zeros((X_in.shape[0], X_in.shape[1], self.hidden_dim))

        for t in range(X_in.shape[0]):
            x = X_in[t]

            for i in range(self.num_layers):
                # LSTM cell calculations
                combined = x @ self.w_in_list[i].T + h[i] @ self.w_hidden_list[i].T + self.bias_list[i]
                gate_size = self.hidden_dim
                input_gate = torch.sigmoid(combined[:, :gate_size])
                forget_gate = torch.sigmoid(combined[:, gate_size:2*gate_size])
                output_gate = torch.sigmoid(combined[:, 2*gate_size:3*gate_size])
                g_cell = torch.tanh(combined[:, 3*gate_size:])

                # Calculate new cell and hidden state for this layer
                c[i] = forget_gate * c[i] + input_gate * g_cell
                h[i] = output_gate * torch.tanh(c[i])
                x = h[i]

            output[t] = h[-1]

        return output, (h, c)
    
class Embedding(torch.nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = embed_dim 
        self.lookup = torch.nn.Parameter(torch.randn((in_dim, embed_dim)))

    def forward(self, input):
        # input ~ (batches, seqlen), (batches, seqlen, 1), or (batches, seqlen, vocab_size)
        if len(input.shape) == 2 or input.shape[-1] == 1:
            return self.lookup[input, :]  
        else: # assume input are onehot vectors 
            return self.lookup[input.argmax(-1), :]  

class LayerNorm(torch.nn.Module): 
    def __init__(self, features, eps=1.0e-5, withbias=True):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(features))
        self.bias = torch.nn.Parameter(torch.zeros(features))
        self.eps=eps
        self.withbias=withbias

    def forward(self, X): 
        # X of size (*, features)
        mean = torch.mean(X, dim=-1, keepdim=True) 
        var = torch.var(X, dim=-1, keepdim=True)
        out = self.scale * (X - mean) / torch.sqrt(var + self.eps) 
        if self.withbias:
            out += self.bias 
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=None, device=None, dtype=None):
        """
        keys, queries, vals ~ (batches, seq_length, embed_dim)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.dropout = dropout 
        self.head_dim = embed_dim // num_heads

        if dropout is not None:
            if dropout < 0: 
                raise ValueError("Negative dropout")
            self.dropoutlayer = Dropout_1d(dropout)

        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by num_heads")

        weight_bound = math.sqrt(3/embed_dim)
        self.in_weights = torch.nn.Parameter(
            torch.empty((3, embed_dim, embed_dim), dtype=dtype, device=device)
            .uniform_(-weight_bound, weight_bound)
        )
        self.out_weights = torch.nn.Parameter(
            torch.empty((embed_dim, embed_dim), dtype=dtype, device=device)
            .uniform_(-weight_bound, weight_bound)
        )

    def forward(self, key, query, value, mask=None):
        """
        input: key, query, value ~ (batches, seq_len, embed_dim)
        output: (batches, seq_len, embed_dim)
        mask: (seq_len, seq_len) populated by ones and zeros, or something that can be 
        broadcast to (batches, heads, seq_len, seq_len). Alternatively, the keyword
        "mask="causal" can be given for causal attention. 
        """
        n_batch = key.shape[0]
        seq_len = key.shape[1]

        # project full qkv's into set of qkv's for the heads
        # each final tensor has size (batches, heads, seq_len, head_dim)
        keys = (key @ self.in_weights[0]).view(n_batch, -1, self.num_heads, self.head_dim).transpose(1,2)
        queries = (query @ self.in_weights[1]).view(n_batch, -1, self.num_heads, self.head_dim).transpose(1,2)
        values = (value @ self.in_weights[2]).view(n_batch, -1, self.num_heads, self.head_dim).transpose(1,2) 

        #compute similarities for all heads: softmax(Q K^T / sqrt{head_dim} )
        #size: (batches, heads, seq_len, seq_len)
        q_dot_k = queries @ keys.transpose(-1, -2) / math.sqrt(self.head_dim)

        if mask=="causal":
            mask = torch.tril(torch.ones((seq_len, seq_len))) #matrix with zeros above diagonal
            q_dot_k = q_dot_k.masked_fill( mask == 0, -float('inf'))
        elif mask is not None:
            q_dot_k = q_dot_k * mask 

        compatibilities = torch.softmax(q_dot_k, -1)
        if self.dropout is not None:
            compatibilities = self.dropoutlayer(compatibilities)

        # matmult -> (batches, heads, seq_len, head_dim)
        # transpose+rehsape -> (batches, seq_len, embed_dim)
        attentions = (compatibilities @ values).transpose(1, 2).reshape(n_batch, seq_len, -1)

        if self.dropout is not None:
            attentions = self.dropoutlayer(attentions)

        return attentions @ self.out_weights

class PositionalEncoder(torch.nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000, dropout=None):
        super().__init__() 
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len 

        pos_encoder = torch.zeros((max_seq_len, embed_dim)) 
        positions = torch.arange(max_seq_len).unsqueeze(1)
        frequency = (10000 ** (- torch.arange(embed_dim) / embed_dim )).unsqueeze(0)
        pos_encoder[:, 0::2] = torch.sin(positions * frequency[:, 0::2])
        pos_encoder[:, 1::2] = torch.cos(positions * frequency[:, 0::2]) 

        self.register_buffer('positional_encoder', pos_encoder)
        self.dropout= Dropout_1d(dropout)

    def forward(self, X): 
        # X: (batch, seq_length, embed_dim)
        seq_len = X.shape[-2] 
        if seq_len > self.max_seq_len:
            raise ValueError("Input sequence length exceeds max sequence length of positional \
                             encoder")
        return X + self.dropout(self.positional_encoder[0:seq_len])

class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, d_ffn, dropout=None, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_ffn = d_ffn

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, device=device, 
                                            dtype=dtype)
        self.layernorm1 = LayerNorm(embed_dim)
        self.layernorm2 = LayerNorm(embed_dim)
        self.linear1 = Linear(embed_dim, self.d_ffn, weight_init_scheme="kaiming", device=device,
                              dtype=dtype)
        self.linear2 = Linear(self.d_ffn, embed_dim, weight_init_scheme="kaiming", device=device,
                              dtype=dtype)
        self.dropout = Dropout_1d(dropout)

    def forward(self, input, mask=None):
        # input: (batch_size, seq_len, embed_dim)
        # output: (batch_size, seq_len, embed_dim)
        # mask="causal" gives a causal mask (no attention to future tokens)
        self_atn = self.self_attention(input, input, input, mask=mask)
        x = self.layernorm1(self_atn + self.dropout(self_atn))
        fnn = self.linear2(functions.relu(self.linear1(x)))
        return self.layernorm2(x + self.dropout(fnn))

class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, d_ffn, dropout=None, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_ffn = d_ffn

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, device=device, 
                                            dtype=dtype)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout, device=device, 
                                            dtype=dtype)
        self.layernorm1 = LayerNorm(embed_dim)
        self.layernorm2 = LayerNorm(embed_dim)
        self.layernorm3 = LayerNorm(embed_dim)
        self.linear1 = Linear(embed_dim, self.d_ffn, weight_init_scheme="kaiming", device=device,
                              dtype=dtype)
        self.linear2 = Linear(self.d_ffn, embed_dim, weight_init_scheme="kaiming", device=device,
                              dtype=dtype)
        self.dropout = Dropout_1d(dropout)

    def forward(self, dec_input, enc_output, mask=None, cross_mask=None):
        # input, input_enc: (batch_size, seq_len, embed_dim)
        # output: (batch_size, seq_len, embed_dim)
        # mask="causal" gives a causal mask (no attention to future tokens)

        self_atn = self.self_attention(dec_input, dec_input, dec_input, mask=mask)
        x = self.layernorm1(dec_input + self.dropout(self_atn))
        cross_atn = self.cross_attention(x, enc_output, enc_output, mask=cross_mask)
        x = self.layernorm2(x + self.dropout(cross_atn))
        fnn = self.linear2(functions.relu(self.linear1(x)))
        return self.layernorm3(x + self.dropout(fnn))
