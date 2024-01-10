import cv2
import pyrealsense2 as rs
import numpy as np
import math
import time
import os
from collections import deque

# w, h = 1280, 720
# w, h = 848, 480
w, h = 640, 360

input_fps = 30

output_dir = 'record'
zeros_color = np.zeros((h, w, 3), dtype=np.uint8)

os.makedirs(output_dir, exist_ok=True)

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

def depth_scale(depth_frame: rs.depth_frame):
	scope_in_meter = (0.5, 1.5) # [meter]

	depth_image = np.asanyarray(depth_frame.get_data())
	dtype = depth_image.dtype

	invalid_area = (depth_image == 0)
	depth_image[invalid_area] -= 1

	depth_unit = depth_frame.get_units() # [meter/bit]
	scope_in_bit =  (scope_in_meter[0] // depth_unit, scope_in_meter[1] // depth_unit) # [bit]

	# Trim out-of-scope values
	depth_image[depth_image < scope_in_bit[0]] = scope_in_bit[0]
	depth_image[depth_image > scope_in_bit[1]] = scope_in_bit[1]

	# Scale
	depth_image = (depth_image - scope_in_bit[0]) * (np.iinfo(dtype).max // (scope_in_bit[1] - scope_in_bit[0]))

	# Histgram equalization
	# depth_image = cv2.equalizeHist(depth_image)

	return depth_image, ~invalid_area


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

def add_mask(image_base: np.ndarray, mask: np.ndarray, color=(0, 0, 255)):
	result = image_base.copy()
	color = np.array(color)

	# mask_color = zeros_color.copy()
	# mask_color[mask] = color
	# alpha = 0.3

	# result[mask] = cv2.addWeighted(mask_color, alpha, result, 1-alpha, 0)[mask]
	# mask_edge = (~mask) & binarize(cv2.morphologyEx(bool2uint8(mask), cv2.MORPH_DILATE, kernel=np.ones((7, 7), np.uint8), iterations = 1), threshold=1)
	# result[mask_edge] = color

	result[mask] = color

	return result

def centroid(img: np.ndarray):
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

	a_center = np.array(centroid(a))
	b_center = np.array(centroid(b))

	diff = b_center - a_center
	diff_a = diff // 2
	diff_b = -diff + diff_a

	a = translate(a, diff_a[0], diff_a[1])
	b = translate(b, diff_b[0], diff_b[1])

	return a, b, diff_a, diff_b

def calc_mask_depth(mask, depth_image, depth_valid_area):
	mask = binarize(mask)

	indices = mask & depth_valid_area
	if indices.any():
		return np.mean(depth_image[indices])
	else:
		return -1

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

def write_info_image(text):
	info_image = zeros_color.copy()
	y = 0
	for line in text.split('\n'):
		y += 30

		color = (255, 255, 255)
		if line:
			if line.startswith('[red]'):
				color = (0, 0, 255)
				line = line[5:]
			elif line.startswith('[blue]'):
				color = (255, 0, 0)
				line = line[6:]
			elif line.startswith('[green]'):
				color = (0, 255, 0)
				line = line[7:]

			cv2.putText(info_image, line, (10, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, thickness=2)

	return info_image

if __name__ == '__main__':
	config = rs.config()
	config.enable_stream(rs.stream.depth, w, h, rs.format.z16, input_fps)
	config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, input_fps)

	aligner = rs.align(rs.stream.color) # align to color
	# aligner = rs.align(rs.stream.depth) # align to depth

	pipeline = rs.pipeline()
	pipeline.start(config)

	color_image = zeros_color
	depth_image = zeros_color
	depth_valid_area = zeros_color

	fps_counter = Fps_Counter()

	is_recording = False
	writer = None

	result_message = ''

	while True:
		frames = pipeline.wait_for_frames()
		frames = aligner.process(frames)
		color_frame = frames.get_color_frame()
		depth_frame = frames.get_depth_frame()
		if not (color_frame and depth_frame):
			continue

		color_image = np.asanyarray(color_frame.get_data())
		depth_image, depth_valid_area = depth_scale(depth_frame)
		depth_image = cv2.convertScaleAbs(depth_image, alpha=1/2**8)

		depth_image = gray2rgb(depth_image)
		depth_valid_area = gray2rgb(bool2uint8(depth_valid_area))

		fps = fps_counter.count()

		info_message = f'{w}x{h} @ {fps:3.2f} fps\n'
		info_message += '\n'
		if is_recording:
			info_message += '[red]Recording...\n'
			info_message += 'Press R to stop recording.\n'
		else:
			info_message += 'Recording not started.\n'
			info_message += 'Press R to start recording.\n'

		info_message += '\n'
		info_message += result_message

		info_image = write_info_image(info_message)

		cv2.imshow('RGBD recorder', xstack([
			color_image
			, depth_image
			, depth_valid_area
			, info_image
		]))

		if is_recording:
			writer.write(np.hstack([
				color_image
				, depth_image
				, depth_valid_area
			]))

		key = cv2.waitKey(1)
		if key == 27: # Esc
			if not is_recording:
				break
		elif key == ord('r'):
			if not is_recording:
				is_recording = True

				output_filename = time.strftime('%Y-%m%d-%H%M%S.mp4')
				output_filepath = os.path.join(output_dir, output_filename)

				writer = cv2.VideoWriter(
					output_filepath,
					cv2.VideoWriter_fourcc(*'mp4v'),
					fps=input_fps,
					frameSize=(w*3, h),
					isColor=True
				)

				result_message = ''

			else:
				is_recording = False
				writer.release()
				writer = None
				result_message = f'Record has been saved to\n[green]    {output_filepath}'
				print(output_filepath)


	pipeline.stop()
	cv2.destroyAllWindows()