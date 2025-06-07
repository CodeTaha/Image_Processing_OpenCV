import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# blurring (detayı azaltır, gürültüyü engeller)
img = cv2.imread("resimler/NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Orjinal")
plt.show()

"""
Ortalama bulanıklaştırma yöntemi:
Filtre matrisi içerisinde kalan piksellerin ortalaması alınır ve merkezi piksele bu değer yazılır
"""

dst2 = cv2.blur(img, ksize = (3,3))

plt.figure()
plt.imshow(dst2)
plt.axis("off")
plt.title("Ortalama Bulaniklastirma")
plt.show()

"""
Gaussian Bulanıklaştırma
"""

gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)

plt.figure()
plt.imshow(gb)
plt.axis("off")
plt.title("Gaussian Bulaniklastirma")
plt.show()

"""
Medyan Bulanıklaştırma:
Filtreleme matrisi içerisinde kalan piksellerden ortanca değere sahip piksel değeri alınır ve merkezi değere yazılır
"""

mb = cv2.medianBlur(img, ksize = 3)

plt.figure()
plt.imshow(mb)
plt.axis("off")
plt.title("Medyan Bulaniklastirma")
plt.show()

# Gürültü ekleme
def gaussianNoise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    
    return noisy

# İçe aktar ve normalize et
img = cv2.imread("resimler/NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Orjinal")
plt.show()

# Gauss Grültülü Resim
gaussianNoisyImage = gaussianNoise(img)

plt.figure()
plt.imshow(gaussianNoisyImage)
plt.axis("off")
plt.title("Gauss Gurultulu Resim")
plt.show()

# Gauss Blur ile Gürültüyü Giderme
gb = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX = 7)

plt.figure()
plt.imshow(gb)
plt.axis("off")
plt.title("Gaussian Bulaniklastirma")
plt.show()

def saltPepperNoise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    noisy = np.copy(image)
    
    # salt (beyaz gürültü) ekleme
    num_salt = int(np.ceil(amount * image.size * s_vs_p))
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[salt_coords[0], salt_coords[1], :] = 255  # 255: beyaz gürültü
    
    # pepper (siyah gürültü) ekleme
    num_pepper = int(np.ceil(amount * image.size * (1 - s_vs_p)))
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy[pepper_coords[0], pepper_coords[1], :] = 0  # 0: siyah gürültü
    
    return noisy.astype(np.uint8)

# Görüntüyü içe aktar
img = cv2.imread("resimler/NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Normalizasyon yapılmadı

# Tuz Karabiber Gürültülü Resim
spImage = saltPepperNoise(img)

plt.figure()
plt.imshow(spImage)
plt.axis("off")
plt.title("Tuz ve Karabiber Gurultulu Resim")
plt.show()

# Medyan Blur ile Gürültüyü Giderme
mb = cv2.medianBlur(spImage, ksize=3)

plt.figure()
plt.imshow(mb)
plt.axis("off")
plt.title("Medyan Bulaniklastirma ile Gurultu Giderme")
plt.show()
