import cv2

cap = cv2.VideoCapture(0)

# yükseklik ve genişliği alma
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Genişlik: ", width)
print("Yükseklik: ", height)

# video kaydedici, fourcc windows için kullanılan çerçeveleri sıkıştırmaya yarayan dört karakterli codec kodu, 20 fps'i ifade eder
writer = cv2.VideoWriter("videolar/video_kaydi.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 20, (width, height))

while True:
    
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    
    # kaydet
    writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
writer.release()
cv2.destroyAllWindows()
    
    