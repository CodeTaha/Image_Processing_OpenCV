import cv2
import matplotlib.pyplot as plt

img = cv2.imread("resimler/sudoku.jpg", 0)

plt.figure()
plt.imshow(img, cmap = "gray")
plt.axis("off")
plt.title("Orjinal")

# x gradyan
sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)

plt.figure()
plt.imshow(sobelx, cmap = "gray")
plt.axis("off")
plt.title("X gradyan")

# y gradyan
sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)

plt.figure()
plt.imshow(sobely, cmap = "gray")
plt.axis("off")
plt.title("Y gradyan")

# laplacian gradyan
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)

plt.figure()
plt.imshow(laplacian, cmap = "gray")
plt.axis("off")
plt.title("Laplacian")