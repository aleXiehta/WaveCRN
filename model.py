import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from sru import SRU

class WaveCRN(nn.Module):
    def __init__(self):
        super(WaveCRN, self).__init__()
        self.net = ConvBSRU(frame_size=96, conv_channels=256, stride=48, num_layers=6, dropout=0.0)

    def forward(self, x):
        return self.net(x)

class ConvBSRU(nn.Module):
    def __init__(self, frame_size, conv_channels, stride=128, num_layers=1, dropout=0.1, rescale=False, bidirectional=True):
        super(ConvBSRU, self).__init__()
        num_directions = 2 if bidirectional else 1
        if stride == frame_size:
            padding = 0
        elif stride == frame_size // 2:
            padding = frame_size // 2
        else:
            print(stride, frame_size)
            raise ValueError(
                'Invalid stride {}. Length of stride must be "frame_size" or "0.5 * "frame_size"'.format(stride))
            
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=conv_channels, 
            kernel_size=frame_size, 
            stride=stride,
            padding=padding,
            bias=False
        )
        self.deconv = nn.ConvTranspose1d(
            in_channels=conv_channels,
            out_channels=1,
            kernel_size=frame_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.outfc = nn.Linear(num_directions * conv_channels, conv_channels, bias=False)
        self.sru = SRU(
            input_size=conv_channels,
            hidden_size=conv_channels,
            num_layers=num_layers,
            dropout=dropout,
            rnn_dropout=0.1,
            layer_norm=True,
            rescale=rescale,
            bidirectional=bidirectional
        )

    def forward(self, x):
        output = self.conv(x) # B,C,D
        output_ = output.permute(2, 0, 1) # D, B, C
        output, _ = self.sru(output_) # D, B, 2C
        output = self.outfc(output) # D, B, C
        #output = output_ * F.sigmoid(output)
        output = output_ * output # D, B, C
        output = output.permute(1, 2, 0) # B, C, D
        output = self.deconv(output)
        #output = self.conv11(output)
        output = torch.tanh(output)

        return output
