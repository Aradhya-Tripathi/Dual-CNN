from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def data(path, batch_size, num_workers):
    data = ImageFolder(path)
    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


