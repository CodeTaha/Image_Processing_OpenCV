import cv2

# Orjinal Resim
img = cv2.imread("resimler/lenna.png")
print("Resim Boyutu: ", img.shape)
cv2.imshow("Orjinal Resim", img)

# Yeniden Boyutlandırılmış Resim
imgResized = cv2.resize(img, (700, 700))
print("Yeniden Boyutlandırılmış Resim Boyutu: ", imgResized.shape)
cv2.imshow("Yeniden Boyutlu Resim", imgResized)

# Kırpık Resim
imgCropped = img[129:383,75:422] # yükseklik, genişlik
print("Kırpık Resim Boyutu: ", imgCropped.shape)
cv2.imshow("Kirpik Resim", imgCropped)

if cv2.waitKey() == ord("q"):
    cv2.destroyAllWindows()