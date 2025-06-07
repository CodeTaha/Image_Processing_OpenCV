import cv2
import time

# video içe aktar
cap = cv2.VideoCapture("videolar/sokak_video.mp4")

print("Genişlik: ", cap.get(3))
print("Yükseklik: ", cap.get(4))

if cap.isOpened() == False:
    print("Hata")
    
while True:
    
    # ret return demektir, eğer return True ise frame yazdırılır
    
    ret, frame = cap.read()
    
    if ret == True:
        time.sleep(0.03) # uyarı: kullanmazsak çok hızlı akar
        
        cv2.imshow("Video", frame)
        
    else:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # video yakalamayı bırakır
cv2.destroyAllWindows() # tüm açık pencereleri kapatır