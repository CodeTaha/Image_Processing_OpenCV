import cv2
import numpy as np

# arka plan oluştur
img = np.ones((512, 512, 3), np.uint8) * 255 # beyaz arka plan
print("Resim Boyutu: ", img.shape)
cv2.imshow("Arka Plan", img)

cv2.waitKey() == ord("q")

# Çizgi Çizmek
cv2.line(img, (0,0), (512, 512), (0,0,255), 3)
cv2.imshow("Kirmizi Cizgili Resim", img)

cv2.waitKey() == ord("q")

# Dikdörtgen Oluşturmak
cv2.rectangle(img, (0,0), (100, 100), (255,0,0), 2)
cv2.imshow("Dikdortgenli Resim", img)

cv2.waitKey() == ord("q")

# İçi Dolu Dikdörtgen
cv2.rectangle(img, (100,0), (200, 100), (255,0,0), cv2.FILLED)
cv2.imshow("Iqci Dolu Dikdortgenli Resim", img)

cv2.waitKey() == ord("q")

# Çember
cv2.circle(img, (400, 400), 80, (0,255,0), 2)
cv2.imshow("Cemberli Resim", img)

cv2.waitKey() == ord("q")

# Dolu Çember
cv2.circle(img, (250, 250), 80, (0,255,0), cv2.FILLED)
cv2.imshow("Dolu Cemberli Resim", img)

cv2.waitKey() == ord("q")

# Metin
cv2.putText(img, "Resim", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
cv2.imshow("Metinli Resim", img)

if cv2.waitKey() == ord("q"):
    cv2.destroyAllWindows()