from FeatureExtraction import FEB
from reconst import finalLayers
import torch 
from torch import nn 

class DudeNet(nn.Module):
    def __init__(self, feb, final):
        super(DudeNet, self).__init__()
        self.feb1, self.feb2 = feb().FeatureExtractionBlock()
        self.finallayer = final()


    def forward(self, image):
        feature1 = self.feb1(image)
        feature2 = self.feb2(image)
        return self.finallayer(feature1, feature2, image)


if __name__ == "__main__":
    model = DudeNet(FEB, finalLayers)
    print(model(torch.randn(1,3,150,150)).shape)
