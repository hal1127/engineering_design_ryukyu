import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def lowpassFilter(img, msize=30):
  dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  rows, cols = img.shape
  crow, ccol = rows // 2 , cols // 2

  # create a mask first, center square is 1, remaining all zeros
  mask = np.zeros((rows, cols, 2), np.uint8)
  mask[crow-msize : crow+msize, ccol-msize : ccol+msize] = 1

  # apply mask and inverse DFT
  fshift = dft_shift * mask
  f_ishift = np.fft.ifftshift(fshift)
  img_back = cv2.idft(f_ishift)
  img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

  # 0~255で正規化
  img_back_max = img_back.max()
  img_back2 = (img_back * 255) / img_back_max
  img_back2 = img_back2.astype(np.uint8)
  return img_back2

def plotImg(noise, imp, org, plt, noimg=False, nopsnr=False):
  matplotlib.rcParams["font.family"] = "Meiryo"
  if not noimg:
    # 画像をプロット
    fig, ax = plt.subplots(1, 3, figsize=(12,5))
    ax[0].imshow(org, cmap="gray"), ax[0].set_title("元画像")
    ax[1].imshow(noise, cmap="gray"), ax[1].set_title("ノイズ付き画像")
    ax[2].imshow(imp, cmap="gray"), ax[2].set_title("ノイズ除去後画像")
    fig.tight_layout()
    fig.show()

  noise_psnr = cv2.PSNR(org, noise)
  imp_psnr = cv2.PSNR(org, imp)
  if not nopsnr:
    # PSNRを表示
    print("noize PSNR =", noise_psnr)
    print("improvement PSNR =", imp_psnr)
    print("diff PSNR = ", imp_psnr-noise_psnr)

  return imp_psnr - noise_psnr


# 入力画像を読み込み
noise = cv2.imread("(ED15).bmp", 0)
org = cv2.imread("road.bmp", 0)

dst1 = lowpassFilter(noise, 59)

dst2 = cv2.bilateralFilter(dst1, 8, 56, 69)

# 画像をプロット
plotImg(noise, dst2, org, plt)