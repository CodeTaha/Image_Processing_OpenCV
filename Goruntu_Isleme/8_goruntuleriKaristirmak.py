import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("C:/Users/Konyar/Desktop/Screenshot_20250418_152804.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread("C:/Users/Konyar/Desktop/Screenshot_20250418_152810.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# aynı boyutta olmak zorundadırlar
print(img1.shape)
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# karıştırılmış resim = alpha*img1 + beta*img2
blended = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma = 0)

plt.figure()
plt.imshow(blended)