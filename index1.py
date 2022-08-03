import cv2

img = cv2.imread(filename="C:/imagem.png")
img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
cv2.imshow('imagem', img)
cv2.waitKey(0)
cv2.destroyAllWindows()