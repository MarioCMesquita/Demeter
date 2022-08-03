import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('webcam', frame)
    
# press escape to exit
    if (cv2.waitKey(30) == 27):
       break
cap.release()
cv2.destroyAllWindows()