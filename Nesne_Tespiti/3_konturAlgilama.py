import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resmi içe aktar
img_path = "resimler/contour.jpg"
img = cv2.imread(img_path, 0)

# Resmin yüklenip yüklenmediğini kontrol et
if img is None:
    raise FileNotFoundError(f"Görsel yüklenemedi! Dosya yolu yanlış olabilir: {img_path}")

plt.figure()
plt.imshow(img, cmap="gray")
plt.axis("off")

# cv2.RETR_CCOMP: İç ve dış konturları bulur
contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Boş maskeler oluştur
external_contour = np.zeros(img.shape, dtype=np.uint8)
internal_contour = np.zeros(img.shape, dtype=np.uint8)

# Konturları çiz
if hierarch is not None:  # Eğer hiyerarşi bilgisi varsa işlem yap
    hierarch = hierarch[0]  # Çok boyutlu diziden ilk kısmı al

    for i in range(len(contours)):
        if hierarch[i][3] == -1:  # Dış konturlar
            cv2.drawContours(external_contour, contours, i, 255, -1)
        else:  # İç konturlar
            cv2.drawContours(internal_contour, contours, i, 255, -1)

# Dış konturları göster
plt.figure()
plt.imshow(external_contour, cmap="gray")
plt.axis("off")
plt.show()

# İç konturları göster
plt.figure()
plt.imshow(internal_contour, cmap="gray")
plt.axis("off")
plt.show()