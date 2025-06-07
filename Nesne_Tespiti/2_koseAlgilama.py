import cv2
import matplotlib.pyplot as plt
import numpy as np

# Orijinal resmi gri tonlamalı (tek kanal) olarak yükleme
img_gray = cv2.imread("resimler/sudoku.jpg", 0)
img_gray = np.float32(img_gray)
print(img_gray.shape)

plt.figure(),
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

# Harris köşe tespiti
dst = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
plt.figure()
plt.imshow(dst, cmap="gray")
plt.axis("off")

dst = cv2.dilate(dst, None)
img_gray[dst > 0.2 * dst.max()] = 1
plt.figure()
plt.imshow(dst, cmap="gray")
plt.axis("off")

# Shi-Tomasi köşe tespiti için resmi RENKLİ yükleme
img_color = cv2.imread("resimler/sudoku.jpg")  # Renkli olarak yükledik
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # BGR'den RGB'ye çevirdik

corners = cv2.goodFeaturesToTrack(np.float32(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)), 120, 0.01, 10)
corners = np.int64(corners)

# Köşeleri çizdirme
for i in corners:
    x, y = i.ravel()
    cv2.circle(img_color, (x, y), 3, (255, 0, 0), cv2.FILLED)

plt.figure()  
plt.imshow(img_color)
plt.axis("off")

plt.show()
