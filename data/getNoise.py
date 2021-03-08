import cv2 
from skimage.util import random_noise
import os

def noise(path, seed=None, noiseType='speckle'):
    im = cv2.imread(path, 1)
    return random_noise(im, mode=noiseType, seed=seed, clip=True)

if __name__ == '__main__':
    # image = noise('/home/aradhya/Desktop/Research-DudeNet/all-images/im0014.ppm', seed=100)
    # cv2.imshow('testing', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    path = '/home/aradhya/Desktop/Research-DudeNet/all-images/'
    trainingPath = '/home/aradhya/Desktop/Research-DudeNet/Training/'

    def saveImage(path, trainingPath):
        if os.path.exists(trainingPath):
            pass

        else:
            os.mkdir(trainingPath)
        print(len(os.listdir(path)))
        for i in os.listdir(path):
            imagePath = path+i
            im = noise(imagePath, None)
            
            if not cv2.imwrite(trainingPath+i, im*255):
                raise Exception('File not saved')

    saveImage(path, trainingPath)

        
        


