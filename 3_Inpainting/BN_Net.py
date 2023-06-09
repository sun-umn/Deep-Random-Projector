import torch.nn as nn
import torch
import torch.nn.functional as F


#-----------------------------------------------------------
#-----------------------------------------------------------
# BN for the input seed 
#-----------------------------------------------------------
#-----------------------------------------------------------
class BNNet(nn.Module):
    def __init__(self,num_channel):
        super(BNNet, self).__init__()
        self.bn = nn.BatchNorm2d(num_channel)

    def forward(self, input_data):
        output_data = self.bn(input_data)
        return output_data
