from torch import nn
import torch
from model.model import DudeNet
from data import getdata
from model.FeatureExtraction import FEB
from model.reconst import finalLayers

path = '/'

device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 1e-3 ##paper values
beta_1 = 0.9 ##paper values
beta_2 = 0.999 ##paper values
batch_size = 128 ##paper values
epochs = 70 ##paper values

dataloader = getdata(path)
model = DudeNet(FEB, finalLayers)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[beta_1, beta_2])

def train():
    epoch_loss = 0
    model.eval()

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        modelout = model(x)
        loss = criterion(modelout, y)
        
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        epoch_loss += loss
            
    return epoch_loss/len(dataloader)

if __name__ == '__main__':

    for epoch in epochs:
        epoch_loss = train()
        print(f'EPOCH: {epoch} || LOSS: {epoch_loss}')