from torchvision import transforms
from PIL import Image
import torch 
import os


def generate(imagePath, num_samples=15, scale_factor=0.7): ##run this fucntion with 0.8, 0.9 as well as the scale factor
    count = 0
    ext = '.ppm'
    if os.path.exists(imagePath):
            print("Folder Found")
            size = len(os.listdir(imagePath))
            print(size)
            
    else:
            raise RuntimeError("folder not found")
    
    for image in os.listdir(imagePath):
        count += 1
        path = imagePath+image 
        tensor = transforms.ToTensor()(Image.open(path)).unsqueeze(0)
        image = torch.nn.functional.interpolate(input=tensor, scale_factor=scale_factor,
                                                recompute_scale_factor=True, mode='bicubic').clamp(min=0, max=255).squeeze(0)
                                                
        transforms.ToPILImage()(image).save(imagePath+str(count)+ext)                                                
        if count > num_samples:
            break
        

generate('/home/aradhya/Desktop/Research-DudeNet/Training/') ## dont forget the trailing slash

