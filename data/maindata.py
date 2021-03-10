from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

class MyData(Dataset):
    def __init__(self, groundTruthpath, trainingpath, std=(0.5,0.5,0.5), mean=(0.5,0.5,0.5), size=(256, 256)):
        print(len(os.listdir(groundTruthpath)), len(os.listdir(trainingpath)))
        # assert len(os.listdir(groundTruthpath)) == len(os.listdir(trainingpath))
        self.xpath, self.ypath = [], []
        for i in os.listdir(groundTruthpath):
            self.ypath.append(groundTruthpath+i)

        for i in os.listdir(trainingpath):
            self.xpath.append(trainingpath+i)

        self.trans = transforms.Compose([
                     transforms.Resize(size),
                     transforms.RandomRotation((90, 270)),
                     transforms.RandomHorizontalFlip(p=0.3),
                     transforms.ToTensor(),
                     transforms.Normalize(std, mean) 
        ])


    def __len__(self):
        return len(self.xpath)

    def __getitem__(self, xid):
        return self.trans(Image.open(self.xpath[xid])), self.trans(Image.open(self.ypath[xid])) ##returns training image, label image


if __name__ == '__main__':
    data = MyData(groundTruthpath='/home/aradhya/Desktop/Research-DudeNet/all-images/', trainingpath='/home/aradhya/Desktop/Research-DudeNet/Training/')
    loader = DataLoader(data, batch_size=1, num_workers=2)

    for x, y in loader:
        print(x.shape, y.shape)
        break