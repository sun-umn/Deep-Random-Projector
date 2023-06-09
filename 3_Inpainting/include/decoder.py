import torch
import torch.nn as nn

#--------------------custom activation function
# simply define a ABS function
def get_abs(input):
    return torch.abs(input)

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class ABS_FC(nn.Module):

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return get_abs(input) # simply apply already implemented SiLU



def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def decodernw(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad ='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        upsample_first = True,
        ):
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales
    model = nn.Sequential()

    
    for i in range(len(num_channels_up)-1):
        # if i==0:
        #     model.add(nn.BatchNorm2d(num_channels_up[i], affine=bn_affine))
        
        if upsample_first:
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad))
            if upsample_mode!='none' and i != len(num_channels_up)-2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        else:
            if upsample_mode!='none' and i!=0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up[i], 1, pad=pad))        
        
        if i != len(num_channels_up)-1:	
            if(bn_before_act): 
                model.add(nn.BatchNorm2d( num_channels_up[i+1] ,affine=bn_affine))
            model.add(act_fun)
            if(not bn_before_act):
                model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
      
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model




#### let's rebuild the deep decoder with ABS activation function

class DDNET(nn.Module):
    def __init__(self,  num_output_channels, num_channels_up):
        super(DDNET, self).__init__()
        self.conv1 = nn.Conv2d(num_channels_up[0], num_channels_up[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn1 = nn.BatchNorm2d(num_channels_up[1])

        self.conv2 = nn.Conv2d(num_channels_up[1], num_channels_up[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn2 = nn.BatchNorm2d(num_channels_up[2])

        self.conv3 = nn.Conv2d(num_channels_up[2], num_channels_up[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn3 = nn.BatchNorm2d(num_channels_up[3])

        self.conv4 = nn.Conv2d(num_channels_up[3], num_channels_up[4], kernel_size=1, stride=1, padding=0, bias=False)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn4 = nn.BatchNorm2d(num_channels_up[4])

        self.conv5 = nn.Conv2d(num_channels_up[4], num_channels_up[4], kernel_size=1, stride=1, padding=0, bias=False)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn5 = nn.BatchNorm2d(num_channels_up[4])

        self.conv6 = nn.Conv2d(num_channels_up[4], num_channels_up[4], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(num_channels_up[4])
        self.conv7 = nn.Conv2d(num_channels_up[4], num_output_channels, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, input_data):
        output_data = self.conv1(input_data)
        output_data = self.up1(output_data)
        ## let's add our abs activation fuction instead of ReLU here
        #output_data = torch.abs(output_data)
        output_data = torch.relu(output_data)
        output_data = self.bn1(output_data)

        output_data = self.conv2(output_data)
        output_data = self.up2(output_data)
        ## let's add our abs activation fuction instead of ReLU here
        #output_data = torch.abs(output_data)
        output_data = torch.relu(output_data)
        output_data = self.bn2(output_data)

        output_data = self.conv3(output_data)
        output_data = self.up3(output_data)
        ## let's add our abs activation fuction instead of ReLU here
        #output_data = torch.abs(output_data)
        output_data = torch.relu(output_data)
        output_data = self.bn3(output_data)

        output_data = self.conv4(output_data)
        output_data = self.up4(output_data)
        ## let's add our abs activation fuction instead of ReLU here
        #output_data = torch.abs(output_data)
        output_data = torch.relu(output_data)
        output_data = self.bn4(output_data)

        output_data = self.conv5(output_data)
        output_data = self.up5(output_data)
        ## let's add our abs activation fuction instead of ReLU here
        #output_data = torch.abs(output_data)
        output_data = torch.relu(output_data)
        output_data = self.bn5(output_data)

        output_data = self.conv6(output_data)
        ## let's add our abs activation fuction instead of ReLU here
        #output_data = torch.abs(output_data)
        output_data = torch.relu(output_data)
        output_data = self.bn6(output_data)

        output_data = self.conv7(output_data)
        #output_data = torch.sigmoid(output_data) # let's delete the sigmod as well

        return output_data












# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, 1, 1, padding=0, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out

def resdecoder(
        num_output_channels=3, 
        num_channels_up=[128]*5, 
        filter_size_up=1,
        need_sigmoid=True, 
        pad='reflection', 
        upsample_mode='bilinear', 
        act_fun=nn.ReLU(), # nn.LeakyReLU(0.2, inplace=True) 
        bn_before_act = False,
        bn_affine = True,
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    model = nn.Sequential()

    for i in range(len(num_channels_up)-2):
        
        model.add( ResidualBlock( num_channels_up[i], num_channels_up[i+1]) )
        
        if upsample_mode!='none':
            model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))	
            #model.add(nn.functional.interpolate(size=None,scale_factor=2, mode=upsample_mode))	
        
        if i != len(num_channels_up)-1:	
            model.add(act_fun)
            #model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=bn_affine))
                
    # new
    model.add(ResidualBlock( num_channels_up[-1], num_channels_up[-1]))
    #model.add(nn.BatchNorm2d( num_channels_up[-1] ,affine=bn_affine))
    model.add(act_fun)
    # end new
    
    model.add(conv( num_channels_up[-1], num_output_channels, 1, pad=pad))
    
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model


