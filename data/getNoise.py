import cv2 
from skimage.util import random_noise
import os

def noise(path, seed=None, noiseType='speckle'):
    im = cv2.imread(path, 1)
    return random_noise(im, mode=noiseType, seed=seed, clip=True)

if __name__ == '__main__':
    # image = noise('./BSR_bsds500/BSR/BSDS500/data/images/train/376020.jpg', seed=100)
    # cv2.imshow('testing', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    path = './BSR_bsds500/BSR/BSDS500/data/images/train/'
    trainingPath = './BSR_bsds500/BSR/BSDS500/data/images/noiseTraining/'

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

        
        


