import cv2
import numpy as np
import os
import time
from tkinter import filedialog

input_dir = r'output'
input_dir = filedialog.askdirectory(initialdir = input_dir)
fps = 24

from util import *

count = 0
t0 = time.time()

frames_rgb = read_frames_from_images(os.path.join(input_dir, 'rgb'))
frames_mask = read_frames_from_images(os.path.join(input_dir, 'mask'), transform=binarize)

frames_masked = [add_mask(f, m) for f, m in zip(frames_rgb, frames_mask)]

for f in frames_masked:
	t = time.time()

	cv2.imshow('', f)

	key = cv2.waitKey(1)
	if key == 32: # space
		while True:
			key = cv2.waitKey(1)
			if key == 32 or key == 27:
				break
	if key == 27: # esc
		break

	delay = 1/fps - (time.time() - t)
	if delay > 0:
		time.sleep(delay)

cv2.destroyAllWindows()