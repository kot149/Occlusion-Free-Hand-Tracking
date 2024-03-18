import time
import numpy as np
import math
import cv2
import os
import joblib
from tqdm import tqdm
import re

from config import *

###############################################################################
# Misc
###############################################################################
from contextlib import contextmanager, nullcontext
@contextmanager
def TimeKeeper(name, print_result=True):
	t = time.time()
	try:
		yield t
	finally:
		if print_result:
			print(f"% {name}: {(time.time() - t)*1000:4.1f} ms", flush=True)

@contextmanager
def DummyContext():
	yield None


from collections import deque
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

def argmax(_list):
	return max(range(len(_list)), key=_list.__getitem__)


###############################################################################
# Image Processing
###############################################################################
zeros_color = np.zeros((h, w, 3), dtype=np.uint8)
zeros_gray = np.zeros((h, w), dtype=np.uint8)
zeros_bool = np.zeros((h, w), dtype=bool)

def replace_backslash(path:str):
	return path.replace('\\', '/')

def read_frames_from_images(input_dir: str, transform=None):
	filepath_list:list = []

	regex = re.compile(r'(\d+)\.png')

	# List (n, n.png)
	for f in os.listdir(input_dir):
		if not os.path.isfile(os.path.join(input_dir, f)):
			continue

		m = regex.match(f)
		if not m:
			continue

		no = int(m.group(1))

		filepath_list.append((no, os.path.join(input_dir, f)))

	filepath_list.sort(key=lambda x: x[0])

	# Check missing frame
	count = filepath_list[0][0]
	for item in filepath_list:
		i = item[0]
		if i != count:
			print(f"Warning: '{item[1]}' is missing.")
			count = i
		count += 1

	filepath_list = [item[1] for item in filepath_list]

	frames = joblib.Parallel(n_jobs=-1)(joblib.delayed(cv2.imread)(f) for f in tqdm(filepath_list))
	if transform:
		frames = [transform(f) for f in frames]
		# frames = joblib.Parallel(n_jobs=-1)(joblib.delayed(transform)(f) for f in frames)

	return frames


def read_frames_from_video(video_path: str, transform=None):
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	progress = tqdm(total=num_frames)
	frames = []
	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			break

		frames.append(frame)
		progress.update()

	if transform:
		frames = [transform(f) for f in frames]
		# frames = joblib.Parallel(n_jobs=-1)(joblib.delayed(transform)(f) for f in frames)

	return frames, fps

def read_frames(path:str):
	if path.endswith('.mp4'):
		return read_frames_from_video(path)
	else:
		return read_frames_from_images(path)

def binarize(img: np.ndarray, threshold = 128):
	if len(img.shape) == 3:
		img = rgb2gray(img)

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

def add_mask(image_base: np.ndarray, mask: np.ndarray, color=(0, 0, 255)):
	mask = binarize(mask, threshold=1)
	image_base = gray2rgb(image_base)
	result = image_base.copy()
	color = np.array(color)

	# mask_color = zeros_color.copy()
	# mask_color[mask] = color
	# alpha = 0.3

	# result[mask] = cv2.addWeighted(mask_color, alpha, result, 1-alpha, 0)[mask]
	# edge = mask_edge(mask, thickness=5)
	# result[edge] = color

	result[mask] = color

	return result

def binaryMorphologyEx(src, op, kernel, **kwargs):
	src = bool2uint8(src)
	res = cv2.morphologyEx(src, op, kernel, **kwargs)
	res = binarize(res, threshold=1)
	if 'dst' in kwargs:
		kwargs['dst'] = binarize(kwargs['dst'], threshold=1)
	return res

def mask_edge(mask:np.ndarray, thickness:int=1):
	kernel_size = thickness+1
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	dilated = binaryMorphologyEx(mask, cv2.MORPH_DILATE, kernel=kernel, iterations = 1)
	edge = dilated & (~mask)
	return edge

def centroid(img: np.ndarray):
	if not img.any():
		return None
	if img.dtype == bool:
		img = img * np.ones((h, w), dtype=np.uint8)
	m = cv2.moments(img)

	return (int(m['m10']/m['m00']), int(m['m01']/m['m00'])) # (x, y)


def translate(img:np.array, dx, dy):
	is_bool = (img.dtype == bool)
	if is_bool:
		img = bool2uint8(img)

	coeff = np.float32([[1, 0, dx], [0, 1, dy]])
	result =  cv2.warpAffine(img, coeff, (w, h))

	if is_bool:
		result = binarize(result)

	return result

def calc_iou(a : np.ndarray, b : np.ndarray):
	# a, b : 2値画像　マスク部分が白
	a = binarize(a, threshold=1)
	b = binarize(b, threshold=1)

	intersection = (a & b).sum()
	union = (a | b).sum()

	iou = intersection / union if union != 0 else 0

	return iou

def calc_iou_fixed(a : np.ndarray, b : np.ndarray):
	a, b, diff1, diff2 = fix_mask(a, b)

	return calc_iou(a, b)

def fix_mask(a : np.ndarray, b : np.ndarray):
	# 重心を基に2つのマスクの位置を合わせる
	# a, b : 2値画像　マスク部分が白
	a = binarize(a, threshold=1)
	b = binarize(b, threshold=1)

	a_center = centroid(a)
	b_center = centroid(b)

	if (a_center is not None) and (b_center is not None):
		a_center = np.array(a_center)
		b_center = np.array(b_center)
		diff = b_center - a_center
		diff_a = diff // 2
		diff_b = -diff + diff_a

		a = translate(a, diff_a[0], diff_a[1])
		b = translate(b, diff_b[0], diff_b[1])
	else:
		diff_a = np.array([0, 0])
		diff_b = np.array([0, 0])

	return a, b, diff_a, diff_b

def calc_mask_depth(mask, depth_image, depth_valid_area):
	mask = binarize(mask)

	indices = mask & depth_valid_area
	if indices.any():
		return np.mean(depth_image[indices])
	else:
		return -1

def mask_depth_stat(mask, depth_image, depth_valid_area, erosion_kernel_size=15):
	if erosion_kernel_size > 0:
		mask = binaryMorphologyEx(mask, cv2.MORPH_ERODE, np.ones((erosion_kernel_size, erosion_kernel_size)), iterations=1)

	indices = mask & depth_valid_area
	if indices.any():
		values = depth_image[indices]
		min = np.min(values)
		max = np.max(values)
		std = np.std(values)
		q = np.percentile(values, [25, 50, 75])
		return {
			'min': min,
			'max': max,
			'1/4': q[0],
			'2/4': q[1],
			'med': q[1],
			'3/4': q[2],
			'std': std
		}
	else:
		return None

def xstack(imgs : list):
	imgs = [(gray2rgb(img) if img is not None else zeros_color) for img in imgs]
	n = len(imgs)
	num_cols = math.ceil(math.sqrt(n))
	num_rows = math.ceil(n / num_cols)

	for _ in range(num_cols * num_rows - n):
		imgs.append(zeros_color)

	rows = []
	i = 0
	for r in range(num_rows):
		rows.append(np.hstack(imgs[i:i+num_cols]))
		i += num_cols

	return np.vstack(rows)