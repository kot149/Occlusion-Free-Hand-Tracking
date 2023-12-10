import cv2

camera_id = 2
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
	print(f"Camera [{camera_id}] is not available.")
	exit(0)

while True:
	ret, frame = cap.read()
	if not ret:
		continue

	cv2.imshow("frame", frame)

	lastkey = cv2.waitKey(1)
	if lastkey == 27:
		cap.release()
		cv2.destroyAllWindows()
		break
	if lastkey == ord("s"):
		cv2.imwrite("frame.png", frame)