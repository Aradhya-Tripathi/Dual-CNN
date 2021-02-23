import torch 
from torch import nn 

class FEB(nn.Module):
    def __init__(self):
        super(FEB, self).__init__()

    def FEBNET1(self, inchannel, outchannel, kernel, stride, padding, dilation):
        block = nn.Sequential(
                    nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel,
                               padding=padding, stride=stride,  dilation=dilation, bias=False),
                    nn.BatchNorm2d(outchannel),
                    nn.ReLU(inplace=True),
        )

        return block

    def FEBNET2(self, inchannel, outchannel, kernel, stride, padding):
        block = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel,kernel_size=kernel, stride=stride,
                            padding=padding, bias=True),

                nn.ReLU(inplace=True),

        )     

        return block
    
    def getitem(self):
        blockFEB1 = self.FEBNET1(3, 64, (3,3), 1, 0, 1)
        blockFEB2 = self.FEBNET2(3, 64, (3,3), 1, 0)

        print('-'*20, 'block1')

        print(blockFEB1)

        print('-'*20, 'block2')

        print(blockFEB2)

if __name__ == "__main__":
    febnet = FEB()
    febnet.getitem()
