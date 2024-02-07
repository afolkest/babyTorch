import math
import torch
from torch import Tensor

# If we have an image of height h, kernel of height kh and h-stride sh, then the indiex of the new 
# image runs from i=0, ..., (h-kh)//sh

def find_same_padding(pixels: int, kernel: int, stride: int):
    """
    The minimal number of pixels (along the relevant dimension) that we must pad with using the
    'same' padding scheme, which requires the output size to be ceil(pixels/stride).
    See the appendix of example2_CNN_CIFAR.ipynb for a careful derivation of this formula.
    """
    return max(0, (math.ceil(pixels/stride)-1)*stride + kernel - pixels)

def relu(x): 
    return (x>0)*x

def softmax(x: Tensor, dim_sum: int) -> Tensor:
    return torch.exp(x)/torch.sum(torch.exp(x), dim_sum, keepdim=True)

def one_hot_encode(y: Tensor, n_classes: int) -> Tensor:
    "(batch, 1) -> (batch, n_classes)"
    pvec = torch.zeros(*y.shape, n_classes, device=y.device)
    return pvec.scatter_(-1, y.unsqueeze(1), float(1))


def cross_entropy(y_predicted: Tensor, y_label: Tensor) -> Tensor:
    """
    Input: probabilities for each class
    ypred: (batchsize, n_classes)
    ylabel: (batchsize, n_classes)
    """
    return torch.mean(torch.einsum('ij, ij -> i', y_label, -torch.log(y_predicted)))
