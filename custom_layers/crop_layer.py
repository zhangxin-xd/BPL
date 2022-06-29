import torch.nn as nn

class FlattenLayer(nn.Module):

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)

import torch
import torch.nn as nn

class MaxLayer(nn.Module):

    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self, a, b):
        return torch.max(a, b)

import torch.nn as nn
import torch.nn.functional as F

class PadLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, pad):
        super(PadLayer, self).__init__()
        self.pad = pad

    def forward(self, input):
        F.pad(input, [self.pad] * 4)
import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init

class ScaleLayer(torch.nn.Module):

    def __init__(self, num_features, use_bias=True):
        super(ScaleLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        init.ones_(self.weight)
        self.num_features = num_features

        if use_bias:
            self.bias = Parameter(torch.Tensor(num_features))
            init.zeros_(self.bias)
        else:
            self.bias = None


    def forward(self, inputs):
        if self.bias is None:
            return inputs * self.weight.view(1, self.num_features, 1, 1)
        else:
            return inputs * self.weight.view(1, self.num_features, 1, 1) + self.bias
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = F.sigmoid(x)
        x = x.repeat(1, 1, inputs.size(2), inputs.size(3))
        return inputs * x
import torch.nn as nn

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0 and self.cols_to_crop == 0:
            return input
        elif self.rows_to_crop > 0 and self.cols_to_crop == 0:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, :]
        elif self.rows_to_crop == 0 and self.cols_to_crop > 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]
