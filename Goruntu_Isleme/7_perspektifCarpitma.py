import cv2
import numpy as np

img = cv2.imread("resimler/kart.png")
cv2.imshow("Orjinal Resim", img)

cv2.waitKey() == ord("q")

width = 400
height = 500

pts1 = np.float32([[202, 0], [1, 474], [542,146], [341, 619]])
pts2 = np.float32([[0,0], [0, height], [width, 0], [width, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

imgOutput = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("Duzenlenmis Resim", imgOutput)

if cv2.waitKey() == ord("q"):
    cv2.destroyAllWindows()