import unittest
import torch
import math
from layers import Convolution_2d 

class TestConvolution2D(unittest.TestCase):

    def test_initialization(self):
        """Test if the layer initializes correctly."""
        in_channels = 3
        out_channels = 2
        in_imgsize = (28, 28)
        kernel = (3, 3)
        conv = Convolution_2d(in_channels, out_channels, in_imgsize, kernel)

        self.assertEqual(conv.weights.shape, (out_channels, in_channels, *kernel))
        self.assertEqual(conv.biases.shape, (out_channels,))
        self.assertEqual((conv.in_height, conv.in_width), in_imgsize)

    def test_forward_pass(self):
        """Test the forward pass with a known input."""
        in_channels = 2
        out_channels = 9
        in_imgsize = (100, 100)
        kernel = (8, 8)
        stride = (3, 3)
        conv = Convolution_2d(in_channels, out_channels, in_imgsize, kernel, stride)
        input_tensor = torch.ones((1, in_channels, *in_imgsize))
        output = conv(input_tensor)

        expected_output_shape = (1, out_channels, math.ceil(in_imgsize[0] / stride[0]), math.ceil(in_imgsize[1] / stride[1]))
        self.assertEqual(output.shape, expected_output_shape)

    def test_padding_same(self):
        """Test if 'same' padding works as expected."""
        in_channels = 1
        out_channels = 1
        in_imgsize = (4, 4)
        kernel = (3, 3)
        stride = (1, 1)
        conv = Convolution_2d(in_channels, out_channels, in_imgsize, kernel, stride, padding_type='same')
        input_tensor = torch.ones((1, in_channels, *in_imgsize))
        output = conv(input_tensor)

        self.assertEqual(output.shape, (1, out_channels, *in_imgsize))  # Output size should match input size for 'same' padding

if __name__ == '__main__':
    unittest.main()


