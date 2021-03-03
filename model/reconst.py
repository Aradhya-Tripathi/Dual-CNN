import torch 
from torch import nn 


class finalLayers(nn.Module):
    def __init__(self, channels=128):
        super(finalLayers, self).__init__()
        self.channels = channels

        self.EB1 = nn.Sequential(   
                    ##feature map concatination from feb1 and feb2 
                    ##128 channels
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
        )
    
        self.CB2 = nn.Conv2d(in_channels=self.channels, out_channels=3, kernel_size=(1,1),
                             stride=1, padding=0)

        self.EB2 = nn.Sequential(
                    nn.BatchNorm2d(6),
                    nn.ReLU(inplace=True)
        )


        self.CB3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(1,1),
                             stride=1, padding=0)


                              

    def forward(self, feb1, feb2, image):
        concat = torch.cat((feb1, feb2), dim=1)
        out = self.EB1(concat)
        out = self.CB2(out)
        
        out = torch.cat((image, out), dim=1)
        
        out = self.EB2(out)
        nosieMapping = self.CB3(out) ## final noise mapping

        return nosieMapping+image  ## residual operation for final output

##testing
if __name__ == "__main__":
    from FeatureExtraction import FEB


    extraction = FEB()
    feb1, feb2 = extraction.FeatureExtractionBlock()

    image = torch.randn(1,3,150,150)

    febout1 = feb1(image)
    febout2 = feb2(image)
    print(febout1.shape, febout2.shape)        
    final = finalLayers()
    finalout = final(febout1, febout2, image)
    print(finalout.shape)
