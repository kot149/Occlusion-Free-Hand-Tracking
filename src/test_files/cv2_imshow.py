import cv2

im = cv2.imread("test.jpg")
cv2.imshow("test", im)

cv2.waitKey(5000)

cv2.destroyAllWindows()