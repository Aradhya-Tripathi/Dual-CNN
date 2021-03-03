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
                      self.FEBBlock1(inchannel=3, outchannel=64, kernel=(3,3), stride=1, padding=1),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=1, dilation=2),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=1, dilation=2),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=1, dilation=2),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=0),
                      self.FEBBlock1(inchannel=64, outchannel=64, kernel=(3,3), stride=1, padding=1, dilation=2),
                      
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

    
    def FeatureExtractionBlock(self):
        sparse, block = self.getfebnet1()
        featureblock1 = nn.Sequential(
            sparse,
            block
        ) 
        commonconv, cb1 = self.getfebnet2()

        featureblock2 = nn.Sequential(
                commonconv,
                cb1
        )
        return featureblock1, featureblock2
        
if __name__ == "__main__":
    febnet = FEB()
    
    FEB1, FEB2 = febnet.FeatureExtractionBlock()


    testimage = torch.randn(1,3, 150, 150)
    print(FEB1(testimage).shape)
    print(FEB2(testimage).shape)


