import cv2
import matplotlib.pyplot as plt

# Ana görüntüyü içe aktar
chos = cv2.imread("resimler/chocolates.jpg", 0)
plt.figure()
plt.imshow(chos, cmap="gray")
plt.axis("off")

# Aranacak görüntü
cho = cv2.imread("resimler/nestle.jpg", 0)
plt.figure()
plt.imshow(cho, cmap="gray")
plt.axis("off")

# ORB tanımlayıcı
orb = cv2.ORB_create()

# Anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# BF Matcher (Brute Force)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Hata düzeltilmiş

# Noktaları eşleştir
matches = bf.match(des1, des2)

# Mesafeye göre sırala
matches = sorted(matches, key=lambda x: x.distance)

# Eşleşen resimleri görselleştirme
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags=2)
plt.imshow(img_match)
plt.axis("off")
plt.title("orb")

# sift
sift = cv2.SIFT_create()

# bf
bf = cv2.BFMatcher()

# anahtar nokta tespiti sift ile
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

matches = bf.knnMatch(des1, des2, k = 2)

guzel_eslesme = []

for match1, match2 in matches: 
    if match1.distance < 0.75 * match2.distance:
        guzel_eslesme.append([match1])
        
plt.figure()
sift_matches = cv2.drawMatchesKnn(cho, kp1, chos, kp2, guzel_eslesme, None, flags=2)
plt.imshow(sift_matches)
plt.axis("off")
plt.title("sift")