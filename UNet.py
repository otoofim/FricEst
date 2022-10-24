import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from UNetBlocks import *
from collections import OrderedDict




class UNet(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, inputDim):
        super(UNet, self).__init__()
        self.inputDim = inputDim

        
        #architecture
        self.DownConvBlock1 = DownConvBlock(input_dim = 3, output_dim = 128, ResLayers = 2, initializers = None, padding = 1)
        self.DownConvBlock2 = DownConvBlock(input_dim = 128, output_dim = 256, ResLayers = 2, initializers = None, padding = 1)
        self.DownConvBlock3 = DownConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, initializers = None, padding = 1)
        self.DownConvBlock4 = DownConvBlock(input_dim = 512, output_dim = 256, ResLayers = 2, initializers = None, padding = 1)

        self.UpConvBlock1 = UpConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, initializers = None, padding = 1)
        self.UpConvBlock2 = UpConvBlock(input_dim = (512 * 2), output_dim = 256, ResLayers = 2, initializers = None, padding = 1)
        self.UpConvBlock3 = UpConvBlock(input_dim = (256 * 2), output_dim = 128, ResLayers = 2, initializers = None, padding = 1)
        self.UpConvBlock4 = UpConvBlock(input_dim = (128 * 2), output_dim = 1, ResLayers = 2, initializers = None, padding = 1)
        
        self.UpConvBlock4 = UpConvBlock(input_dim = (128 * 2), output_dim = 1, ResLayers = 2, initializers = None, padding = 1)
        
        self.affine = nn.Sequential(OrderedDict([
          ('flatten', nn.Flatten()),
          ('affine1', nn.Linear(self.inputDim*self.inputDim, 1024)),
          ('relu1', nn.ReLU()),
          ('affine2', nn.Linear(1024, 128)),
          ('relu2', nn.ReLU()),
          ('affine3', nn.Linear(128, 1)),
        ]))

        
    def forward(self, inputFeatures):
        
        encoderOuts = {}
        
        encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
        encoderOuts["out2"] = self.DownConvBlock2(encoderOuts["out1"])
        encoderOuts["out3"] = self.DownConvBlock3(encoderOuts["out2"])
        encoderOuts["out4"] = self.DownConvBlock4(encoderOuts["out3"])

        out = self.UpConvBlock1(encoderOuts["out4"])
        out = torch.cat((encoderOuts["out3"], out), 1)

        out = self.UpConvBlock2(out)
        out = torch.cat((encoderOuts["out2"], out), 1)

        out = self.UpConvBlock3(out)
        out = torch.cat((encoderOuts["out1"], out), 1)
        
        out = self.UpConvBlock4(out)
        out = self.affine(out)
        
        return out
    
    def inference(self, inputFeatures):
        
        with torch.no_grad():
            encoderOuts = {}

            encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
            encoderOuts["out2"] = self.DownConvBlock2(encoderOuts["out1"])
            encoderOuts["out3"] = self.DownConvBlock3(encoderOuts["out2"])
            encoderOuts["out4"] = self.DownConvBlock4(encoderOuts["out3"])

            out = self.UpConvBlock1(encoderOuts["out4"])
            out = torch.cat((encoderOuts["out3"], out), 1)

            out = self.UpConvBlock2(out)
            out = torch.cat((encoderOuts["out2"], out), 1)

            out = self.UpConvBlock3(out)
            out = torch.cat((encoderOuts["out1"], out), 1)

            out = self.UpConvBlock4(out)
            self.affine(out)

            return out
    