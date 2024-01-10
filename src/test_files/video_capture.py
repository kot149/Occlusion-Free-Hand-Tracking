import cv2
import numpy as np
import time
from collections import deque

camera_id = 2
# cap = cv2.VideoCapture(camera_id)
cap = cv2.VideoCapture('record/2023-1216-142823.mp4')

class Fps_Counter:
	def __init__(self, cache_size=30):
		self.fps = -1
		self.cache_size = cache_size
		self.init()

	def init(self):
		self.__cache = deque(maxlen=self.cache_size)
		self.__timestamp = time.time()
		self.fps = 0


	def count(self):
		t = time.time()
		self.__cache.append(t - self.__timestamp)
		self.__timestamp = t

		ave = np.mean(self.__cache)
		self.fps = 1 / ave if ave != 0 else 0

		return self.fps


if not cap.isOpened():
	print(f"Camera [{camera_id}] is not available.")
	exit(0)

fps = cap.get(cv2.CAP_PROP_FPS)
seconds_per_frame = 1/fps

print(fps)

fps_counter = Fps_Counter()

while True:
	t0 = time.time()
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

	print(fps_counter.count())

	delay = seconds_per_frame - (time.time() - t0)
	if delay > 0:
		time.sleep(delay)