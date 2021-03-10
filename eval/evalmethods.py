import skimage

def psnr(img1, img2):
    psnr = skimage.metrics.peak_signal_noise_ratio(img1, img2)
    return psnr

def ssim(img1, img2):
    ssim = skimage.metrics.structural_similarity(img1, img2)
    return ssim ##returns mean ssim