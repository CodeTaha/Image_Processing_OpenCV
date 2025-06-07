import cv2
import matplotlib.pyplot as plt
import numpy as np

# Orijinal görüntüyü yükle
img = cv2.imread("resimler/hist_equ.jpg", 0)
cv2.imshow("Orijinal", img)

# Orijinal görüntünün histogramını hesapla
img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Orijinal histogramı çubuk grafik olarak çiz
plt.figure()
x = np.arange(256)  # Piksel değerleri (0-255)
plt.bar(x, img_hist.flatten(), color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Piksel Değerleri')
plt.ylabel('Genlik')
plt.title('Orijinal Histogram Grafiği')
plt.show()

# Histogram eşitleme
eq_hist_img = cv2.equalizeHist(img)
cv2.imshow("Çıktı", eq_hist_img)

# Eşitlenmiş görüntünün histogramını hesapla
eq_hist = cv2.calcHist([eq_hist_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Eşitlenmiş histogramı çubuk grafik olarak çiz
plt.figure()
plt.bar(x, eq_hist.flatten(), color='green', alpha=0.7, edgecolor='black')  # Farklı renk seçildi
plt.xlabel('Piksel Değerleri')
plt.ylabel('Genlik')
plt.title('Eşitlenmiş Histogram Grafiği')
plt.show()

if cv2.waitKey() == ord("q"):
    cv2.destroyAllWindows()
