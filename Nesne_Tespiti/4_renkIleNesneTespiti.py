import cv2
import numpy as np
from collections import deque

# Nesne merkezini depolayacak veri tipi
buffer_size = 16
pts = deque(maxlen=buffer_size)

# Mavi renk aralığı HSV
blueLower = (90, 50, 50)
blueUpper = (130, 255, 255)

# Kamera
cap = cv2.VideoCapture(0)
cap.set(3, 1600)
cap.set(4, 900)

while True:
    returnVal, imgOriginal = cap.read()

    if returnVal:
        # Blur
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0)

        # HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)

        # Mavi renk için maske oluşturma
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("Mask Image", mask)

        # Maskenin etrafında kalan gürültüleri sil
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("Erezyon Ve Genisleme", mask)

        # Kontur bulma (düzeltilmiş satır)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            # En büyük kontur'u al
            c = max(contours, key=cv2.contourArea)

            # Dikdörtgen ile çevrele
            rect = cv2.minAreaRect(c)

            ((x, y), (width, height), rotation) = rect

            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(
                np.round(x), np.round(y), np.round(width), np.round(height), np.round(rotation)
            )
            print(s)

            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # Moment hesaplama
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = (0, 0)

            # Konturu çizdir
            cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)

            # Merkeze pembe nokta ekleme
            cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)

            # Bilgileri ekrana yazdır
            cv2.putText(imgOriginal, s, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        # Deque kullanarak nesnenin hareketini takip et
        pts.appendleft(center)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue

            cv2.line(imgOriginal, pts[i - 1], pts[i], (0, 255, 0), 3)

        cv2.imshow("Orjinal Tespit", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()