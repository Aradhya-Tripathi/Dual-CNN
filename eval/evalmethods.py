from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as p


def getpsnr(img1, img2):
    return p(img1, img2)
    

def getssim(img1, img2):
    return ssim(img1, img2, multichannel=True)  ##returns mean ssim

