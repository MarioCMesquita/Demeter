import cv2
import numpy as np
from PIL import ImageGrab

previousFrame = None
cap = cv2.VideoCapture(0)

while True:
    # 1. Load image; convert to RGB
    ret, imgRgb = cap.read()
    # imgRgb = cv2.cvtColor(src=imgBrg, code=cv2.COLOR_BGR2RGB)

    # 2. Prapare image; grayscale and blur
    preparedFrame = cv2.cvtColor(imgRgb, cv2.COLOR_BGR2GRAY)
    preparedFrame = cv2.GaussianBlur(src=preparedFrame, ksize=(5,5), sigmaX=0)

    # 3. Set previous frame and continue if there is None
    if (previousFrame is None):
        # First frame; there is no previous one yet
        previousFrame = preparedFrame
        continue
    
    # calculate difference and update previous frame
    diffFrame = cv2.absdiff(src1=previousFrame, src2=preparedFrame)
    previousFrame = preparedFrame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5,5))
    diffFrame = cv2.dilate(diffFrame, kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    threshFrame = cv2.threshold(src=diffFrame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

    # Contorna a area dos objetos em movimento
    contours, _ = cv2.findContours(image=threshFrame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 50:
            # too small: skip!
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img=imgRgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    cv2.imshow('Motion detector', imgRgb)

    if (cv2.waitKey(30) == 27):
        break

cap.release()
cv2.destroyAllWindows()