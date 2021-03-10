from torch import nn
import torch
from model.model import DudeNet
from model.FeatureExtraction import FEB
from model.reconst import finalLayers
from tqdm.auto import tqdm
from data.maindata import MyData
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

groundTruthpath = '/home/aradhya/Desktop/Research-DudeNet/all-images/' ## trailing slash req
trainingpath = '/home/aradhya/Desktop/Research-DudeNet/Training/' ## trailing slash req

device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 1e-3 ##paper values
beta_1 = 0.9 ##paper values
beta_2 = 0.999 ##paper values
batch_size = 128 ##paper values
epochs = 70 ##paper values


data = MyData(groundTruthpath, trainingpath, size=(150,150)) 

dataloader = DataLoader(data, batch_size=batch_size, num_workers=2)


model = DudeNet(FEB, finalLayers)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=[beta_1, beta_2])

def train(model=model, train_dl=dataloader, loss_fn=criterion, optim=optimizer):
    epoch_loss = 0
    model.train()

    for x, y in tqdm(train_dl, total=len(train_dl), leave=False):
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()

        model_out = model(x)
        
        loss = loss_fn(model_out, y)

        loss.backward()

        optim.step()

        epoch_loss += loss.item()
    
    return epoch_loss/len(train_dl)

if __name__ == "__main__":
    
    for epoch in range(epochs):
        loss = train()
        print(f"EPOCH: {epoch} || LOSS: {loss}")
