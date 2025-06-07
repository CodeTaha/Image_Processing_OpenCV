import cv2

# içe aktarma
img = cv2.imread("resimler/indir.jpg", 0)

# gri tonlamada görüntüleme
cv2.imshow("Cicekler", img)

k = cv2.waitKey(0) & 0xFF

if k == ord('q'):
    cv2.destroyAllWindows()

elif k == ord('s'):
    cv2.imwrite("resimler/cicekler_gray.png", img)
    cv2.destroyAllWindows()
