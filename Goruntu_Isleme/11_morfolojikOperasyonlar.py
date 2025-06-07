import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("resimler/NEURALPNET.png", 0)

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Orjinal")

"""
Erezyon:
Bir piksel matrisi görüntü üzerinde dolaşacak ve dolaşırken ön plandaki nesnenin sınırlarını aşındırır.
"""
kernel = np.ones((3,3), dtype = np.uint8)
erezyon = cv2.erode(img, kernel, iterations = 1)

plt.figure()
plt.imshow(erezyon, cmap="gray")
plt.axis("off")
plt.title("Erezyon")

"""
Genişleme:
Erezyonun tam tersidir, nesnenin sınırlarını genişletir.
"""
dilation = cv2.dilate(img, kernel, iterations = 2)

plt.figure()
plt.imshow(dilation, cmap="gray")
plt.axis("off")
plt.title("Genişleme")

# white noise
whiteNoise = np.random.randint(0, 2, size = img.shape[:2])
whiteNoise = whiteNoise*255

plt.figure()
plt.imshow(whiteNoise, cmap = "gray")
plt.axis("off")
plt.title("White Noise")

white_noise_img = whiteNoise + img

plt.figure()
plt.imshow(white_noise_img, cmap = "gray")
plt.axis("off")
plt.title("Beyaz gürültülü resim")

"""
Açılma:
Beyaz gürültüyü azaltmada kullanılır.
"""
opening = cv2.morphologyEx(white_noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel, iterations = 2)

plt.figure()
plt.imshow(opening, cmap = "gray")
plt.axis("off")
plt.title("Açılma")

# black noise
blackNoise = np.random.randint(0, 2, size = img.shape[:2])
blackNoise = blackNoise*-255

plt.figure()
plt.imshow(blackNoise, cmap = "gray")
plt.axis("off")
plt.title("Black Noise")

black_noise_img = blackNoise + img
black_noise_img[black_noise_img <= -245] = 0

plt.figure()
plt.imshow(black_noise_img, cmap = "gray")
plt.axis("off")
plt.title("Siyah gürültülü resim")

"""
Kapatma:
Siyah gürültüyü azaltmada kullanılır.
"""
closing = cv2.morphologyEx(black_noise_img.astype(np.float32), cv2.MORPH_CLOSE, kernel, iterations = 2)

plt.figure()
plt.imshow(closing, cmap = "gray")
plt.axis("off")
plt.title("Kapatma")

"""
Morfolojik Gradyan:
Nesnelerin iç kısımları aşındırılır.
"""
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.figure()
plt.imshow(gradient, cmap = "gray")
plt.axis("off")
plt.title("Morfolojik Gradyan")