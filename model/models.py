import torch 
from torch import nn 

class FEB(nn.Module):
    def __init__(self):
        super(FEB, self).__init__()

    def FEBBlock1(self, inchannel, outchannel, kernel, stride, padding, dilation=1):
        block = nn.Sequential(
                    nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel,
                               padding=padding, stride=stride,  dilation=dilation, bias=False),
                    nn.BatchNorm2d(outchannel),
                    nn.ReLU(inplace=True),
        )

        return block

    def FEBBlock2(self, inchannel, outchannel, kernel, stride, padding):
        block = nn.Sequential(
                nn.Conv2d(in_channels=inchannel, out_channels=outchannel,kernel_size=kernel, stride=stride,
                            padding=padding, bias=True),

                nn.ReLU(inplace=True),

        )     

        return block
    
    def getfebnet1(self):
        self.sparse = nn.Sequential(
                      self.FEBBlock1(inchannel=3, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0, dilation=2),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0, dilation=2),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0, dilation=2),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0, dilation=2),
                      
        )

        self.febn1 = nn.Sequential(
                    self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                    self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                    self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        )

        return self.sparse, self.febn1


    def getfebnet2(self):
        self.commonconv = nn.Sequential(
                        self.FEBBlock2(inchannel=3, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),
                        self.FEBBlock2(inchannel=64, outchannel=64, kernel=(3,3), stride=1,padding=0),

        )
        

        self.CB1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=1, padding=0)

        return self.commonconv, self.CB1


if __name__ == "__main__":
    febnet = FEB()
    sparse, block = febnet.getfebnet1()
    commonconv, cb1 = febnet.getfebnet2()

    FEB1 = nn.Sequential(
            sparse,
            block
    )

    FEB2 = nn.Sequential(
            commonconv,
            cb1
    )

    # print(FEB1)
    # print('-'*20)
    # print(FEB2)

    testimage = torch.randn(1,3, 250, 250)
    FEB1(testimage)
    FEB2(testimage)

