# babyTorch
This repo is an implementation various deep learning architectures and methods 
from scratch, with the goal of getting an intimate understanding of their inner
workings. The implementation relies of the `torch.Tensor` and `torch.nn.Module` classes 
from PyTorch to handle automatic differentiation for backpropagation, but otherwise the important
elements of the architectures and methods are implemented from scratch. 

What is implemented so far: 
  - Adam and SGD optimization
  - Linear layers
  - Dropout layers (1d)
  - Convolutional layers (2d)
  - Maxpool layers (2d)
  - Long short-term memory (LSTM)

There are currently also three included examples training a MLP, a CNN, and a
LSTM. 

Currently under implementation:
  - Transformer
  - Training of Bach chorale generator using LSTM 

<!--
# Bach dataset: https://github.com/ageron/handson-ml2/blob/master/datasets/jsb_chorales/jsb_chorales.tgz
-->
