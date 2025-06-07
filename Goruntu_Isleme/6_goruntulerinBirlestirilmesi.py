import cv2
import numpy as np

# resmi içe aktar
img = cv2.imread("resimler/lenna.png")
cv2.imshow("Orjinal Resim", img)

horizontal = np.hstack((img, img))
cv2.imshow("Yatay Birlestirilmis Resim", horizontal)

vertical = np.vstack((img, img))
cv2.imshow("Dikey Birlestirilmis Resim", vertical)

if cv2.waitKey() == ord("q"):
    cv2.destroyAllWindows()