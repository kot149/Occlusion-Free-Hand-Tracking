import cv2
import numpy as np
import os
import time
from tkinter import filedialog

input_dir = r'output'
input_dir = filedialog.askdirectory(initialdir = input_dir)
fps = 24

def add_mask(image_base: np.ndarray, mask: np.ndarray, color=(0, 0, 255)):

	result = image_base.copy()
	color = np.array(color)

	# mask_color = zeros_color.copy()
	# mask_color[mask] = color
	# alpha = 0.3

	# result[mask] = cv2.addWeighted(mask_color, alpha, result, 1-alpha, 0)[mask]
	# mask_edge = (~mask) & binarize(cv2.morphologyEx(bool2uint8(mask)	, cv2.MORPH_DILATE, kernel=np.ones((7, 7), np.uint8), iterations = 1), threshold=1)
	# result[mask_edge] = color

	result[mask] = color

	return result

def binarize(img: np.ndarray, threshold = 128):
	if img.dtype == bool:
		return img

	img = img.astype(np.uint8)

	return img >= threshold if threshold >= 0 else img <= -threshold

def bool2uint8(img: np.ndarray):
	h, w = img.shape
	return img * np.ones((h, w), dtype=np.uint8) * 255

def gray2rgb(img: np.ndarray):
	if len(img.shape) != 2:
		return img
	if img.dtype == bool:
		img = bool2uint8(img)
	return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def rgb2gray(img: np.ndarray):
	if len(img.shape) != 3:
		return img
	if img.dtype == bool:
		img = bool2uint8(img)
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def rgb2bgr(img: np.ndarray):
	if len(img.shape) != 3:
		return img
	if img.dtype == bool:
		img = bool2uint8(img)
	return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr2rgb(img: np.ndarray):
	if len(img.shape) != 3:
		return img
	if img.dtype == bool:
		img = bool2uint8(img)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

count = 0
t0 = time.time()
while True:
	t = time.time()
	count += 1
	filename = f'{count:0>5}.png'
	filename_rgb = os.path.join(input_dir, 'rgb', filename)
	filename_mask = os.path.join(input_dir, 'mask', filename)

	if os.path.isfile(filename_rgb) and os.path.isfile(filename_mask):
		rgb_image = cv2.imread(filename_rgb)
		mask_image = cv2.imread(filename_mask)
	else:
		count = 0
		print(time.time() - t0)
		continue


	cv2.imshow('', np.hstack([add_mask(rgb_image, binarize(rgb2gray(mask_image)))]))

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