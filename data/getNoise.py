import cv2 
from skimage.util import random_noise


def noise(path, seed=None):
    im = cv2.imread(path, 1)
    return random_noise(im, mode='speckle', seed=seed, clip=True)

if __name__ == '__main__':
    image = noise('download.jpeg')
    cv2.imshow('testing', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
