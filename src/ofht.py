import torch
from PIL import Image
import numpy as np
import cv2

import math
import time
import sys, os
import ffmpeg
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Manager
from collections import deque


###############################################################################
# Config
###############################################################################

# w, h = 1280, 720
# w, h = 848, 480
w, h = 640, 360

input_fps = 60

input_from_file = False
input_filepath = r'record\2023-1219-141409.mp4'

record_in_video_cv2 = False
record_in_video_ffmpeg = False
record_in_images = False

device = torch.device("cuda")

input_seconds_per_frame = 1 / input_fps

###############################################################################
# Misc
###############################################################################
from contextlib import contextmanager
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

def init_process():
	pass

###############################################################################
# Image Processing
###############################################################################
zeros_color = np.zeros((h, w, 3), dtype=np.uint8)
zeros_gray = np.zeros((h, w), dtype=np.uint8)
zeros_bool = np.zeros((h, w), dtype=bool)

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
	mask = binarize(mask, threshold=1)
	image_base = gray2rgb(image_base)
	result = image_base.copy()
	color = np.array(color)

	mask_color = zeros_color.copy()
	mask_color[mask] = color
	alpha = 0.3

	result[mask] = cv2.addWeighted(mask_color, alpha, result, 1-alpha, 0)[mask]
	edge = mask_edge(mask, thickness=5)
	result[edge] = color

	# result[mask] = color

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
		median = np.median(values)
		std = np.std(values)
		return min, max, median, std
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

###############################################################################
# RGBD Streaming
###############################################################################
import pyrealsense2 as rs

def depth_scale(depth_frame: rs.depth_frame):
	scope_in_meter = (0.3, 1.5) # [meter]

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

def rgbd_streaming_task(shm_rgbd, shm_flags):
	if input_from_file:
		cap = cv2.VideoCapture(input_filepath)

		if cap.isOpened():
			w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 3)
			h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps_v = cap.get(cv2.CAP_PROP_FPS)
			seconds_per_frame_v = 1/fps_v

			print(w_v, h_v)

		fps_counter = Fps_Counter()

		while cap.isOpened() and not shm_flags['end_flag']:
			t0 = time.time()
			ret, frame = cap.read()
			if ret:
				color_image = frame[:, w_v*0:w_v*1, :]
				depth_image = frame[:, w_v*1:w_v*2, :]
				depth_valid_area = frame[:, w_v*2:w_v*3, :]

				color_image = cv2.resize(color_image, (w, h))
				depth_image = cv2.resize(depth_image, (w, h))
				depth_valid_area = cv2.resize(depth_valid_area, (w, h))

				depth_image = rgb2gray(depth_image)
				depth_valid_area = binarize(rgb2gray(depth_valid_area))


				shm_rgbd['color_image'] = color_image
				shm_rgbd['depth_image'] = depth_image
				shm_rgbd['depth_valid_area'] = depth_valid_area
				shm_rgbd['frame_no'] += 1

				shm_rgbd['fps'] = fps_counter.count()

				delay = seconds_per_frame_v - (time.time() - t0)
				if delay > 0:
					time.sleep(delay)

			else:
				break

		cap.release()
	else:
		config = rs.config()
		config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, input_fps)
		config.enable_stream(rs.stream.depth, w, h, rs.format.z16, input_fps)
		# config.enable_record_to_file('d455data.bag')

		# colorizer = rs.colorizer()
		aligner = rs.align(rs.stream.color) # align to color
		# aligner = rs.align(rs.stream.depth) # align to depth

		pipeline = rs.pipeline()
		pipeline.start(config)

		fps_counter = Fps_Counter()
		while not shm_flags['end_flag']:
			frames = pipeline.wait_for_frames()
			frames = aligner.process(frames)
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()

			if not (color_frame and depth_frame):
				continue

			color_image = np.asanyarray(color_frame.get_data())
			depth_image, depth_valid_area = depth_scale(depth_frame)
			depth_image = cv2.convertScaleAbs(depth_image, alpha=1/2**8)

			shm_rgbd['color_image'] = color_image
			shm_rgbd['depth_image'] = depth_image
			shm_rgbd['depth_valid_area'] = depth_valid_area
			shm_rgbd['frame_no'] += 1

			shm_rgbd['fps'] = fps_counter.count()

		pipeline.stop()

	shm_flags['rgbd_streaming_task_closed'] = True
	print("* RGBD Streaming Task Closed")



###############################################################################
# Track-Anything
###############################################################################
from track_anything.tracker.base_tracker import BaseTracker
xmem_checkpoint = 'model_checkpoint/XMem-s012.pth'


###############################################################################
# MediaPipe
###############################################################################
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_landmarks(image, multi_hand_landmarks):
	image = image.copy()

	if multi_hand_landmarks:
		# Draw only one hand
		mp_drawing.draw_landmarks(
			image,
			multi_hand_landmarks[-1],
			mp_hands.HAND_CONNECTIONS,
			mp_drawing_styles.get_default_hand_landmarks_style(),
			mp_drawing_styles.get_default_hand_connections_style()
		)

		# # Draw multiple hands
		# for hand_landmarks in multi_hand_landmarks:
		# 	mp_drawing.draw_landmarks(
		# 		image,
		# 		hand_landmarks,
		# 		mp_hands.HAND_CONNECTIONS,
		# 		mp_drawing_styles.get_default_hand_landmarks_style(),
		# 		mp_drawing_styles.get_default_hand_connections_style()
		# 	)

	return image

def get_landmark_coords(image, multi_hand_landmarks):
	result = []
	h, w, _ = image.shape

	if multi_hand_landmarks:
		for landmark in multi_hand_landmarks[-1].landmark:
			x = int(landmark.x * w)
			y = int(landmark.y * h)

			result.append((x, y))

	return result

def calc_landmark_presition(mask_img, landmark_coords):
	mask_img = binarize(mask_img, threshold=1)
	landmarks_count = 0
	result = 0

	for coord in landmark_coords:
		landmarks_count += 1

		x = coord[0]
		y = coord[1]

		if x < 0 or x >= w or y < 0 or y >= h:
			continue

		result += mask_img[y, x]

	result = result / landmarks_count
	return result

# hand_bbox_size_adjust_count_max = 10
def mediapipe_task(shm_rgbd, shm_mediapipe, shm_flags):
	with mp_hands.Hands(
		model_complexity=0,
		min_detection_confidence=0.2,
		min_tracking_confidence=0.8,
		# num_hands=1
	) as hands:
		tracker = BaseTracker(xmem_checkpoint, device)
		mask_hand = None
		tracker_initialized = False

		frame_no = 0
		fps_counter = Fps_Counter()
		while not shm_flags['end_flag']:
			# Wait for a new frame
			frame_no_prev = frame_no
			frame_no = shm_rgbd['frame_no']
			while not (frame_no > frame_no_prev):
				if shm_flags['end_flag']:
					break
				frame_no = shm_rgbd['frame_no']

			color_image = shm_rgbd['color_image']
			depth_image = shm_rgbd['depth_image']
			depth_valid_area = shm_rgbd['depth_valid_area']

			mp_result = hands.process(bgr2rgb(color_image))

			# When MediaPipe succeeded
			if mp_result and mp_result.multi_hand_landmarks:
				landmark_coords = get_landmark_coords(color_image, mp_result.multi_hand_landmarks)
				shm_mediapipe['coords'] = landmark_coords
				landmark_coords = np.array(landmark_coords)

				"""
				hand_bbox_size_adjust_count = shm_mediapipe['hand_bbox_size_adjust_count']
				if hand_bbox_size_adjust_count < hand_bbox_size_adjust_count_max:

					# Check if all coords are valid
					coords = landmark_coords[[0, 1, 5, 9, 13, 17]]
					for _coord in coords:
						if not depth_valid_area[_coord[1], _coord[0]]:
							break

					else: # All coords are valid
						coords = coords / np.arctan(coords / depth_image[coords[0, 1], coords[0, 0]])

						# mean = np.mean(coords, axis=0)
						std = np.std(coords, axis=0, ddof=1)
						hand_bbox_size = np.mean(std) * 13

						shm_mediapipe['hand_bbox_size'] = shm_mediapipe['hand_bbox_size'] + hand_bbox_size

						hand_bbox_size_adjust_count += 1
						shm_mediapipe['hand_bbox_size_adjust_count'] = hand_bbox_size_adjust_count

						if hand_bbox_size_adjust_count == hand_bbox_size_adjust_count_max:
							shm_mediapipe['hand_bbox_size'] = shm_mediapipe['hand_bbox_size'] / hand_bbox_size_adjust_count_max

							shm_mediapipe['ready'] = True
				"""



				# if mp_result.palm_detections:
				# 	rbb = mp_result.palm_detections[-1].location_data.relative_bounding_box
				# 	x1, y1, x2, y2 = rbb.xmin, rbb.ymin, rbb.width, rbb.height
				# 	x2, y2 = x1+x2, y1+y2
				# 	x1 = int(x1 * w)
				# 	x2 = int(x2 * w)
				# 	y1 = int(y1 * h)
				# 	y2 = int(y2 * h)
				# else:
				x1 = np.min(landmark_coords[:, 0])
				y1 = np.min(landmark_coords[:, 1])
				x2 = np.max(landmark_coords[:, 0])
				y2 = np.max(landmark_coords[:, 1])

				# crop_margin = 0.25
				# margin_x = int((x2 - x1) * crop_margin)
				# margin_y = int((y2 - y1) * crop_margin)
				# x1 -= margin_x
				# x2 += margin_x
				# y1 -= margin_y
				# y2 += margin_y

				x1 = max(x1, 0)
				y1 = max(y1, 0)
				x2 = min(x2, w-1)
				y2 = min(y2, h-1)

				hand_center = ((x1 + x2) // 2, (y1 + y2) // 2)

				if mask_hand is None or shm_flags['reset']:
					half_hand_bbox_size = int((max(x2 - x1, y2 - y1) * 1.3) / 2)
					x1 = hand_center[0] - half_hand_bbox_size
					x2 = hand_center[0] + half_hand_bbox_size
					y1 = hand_center[1] - half_hand_bbox_size
					y2 = hand_center[1] + half_hand_bbox_size

					x1 = max(x1, 0)
					y1 = max(y1, 0)
					x2 = min(x2, w-1)
					y2 = min(y2, h-1)

					color_image_crop = color_image[y1:y2, x1:x2, :]

					everything_masks = do_fastsam(color_image_crop)

					# Revert cropping
					for i, mask in enumerate(everything_masks):
						tmp = zeros_bool.copy()
						tmp[y1:y2, x1:x2] = mask
						everything_masks[i] = tmp


					presitions = [calc_landmark_presition(mask, landmark_coords) for mask in everything_masks]
					presitions_argmax = argmax(presitions)
					mask_hand = everything_masks[presitions_argmax]

					mask_hand2, prob, _ = tracker.track(color_image, mask_hand)
					tracker_initialized = True
					shm_mediapipe['ready'] = True
					shm_flags['reset'] = False

			if tracker_initialized:
				mask_hand, prob, _ = tracker.track(color_image)
				mask_hand = binarize(mask_hand, threshold=1)

			if mask_hand is not None:
				shm_mediapipe['mask_hand'] = mask_hand

			shm_mediapipe['hand_center'] = hand_center

			color_image_with_landmarks = color_image.copy()
			color_image_with_landmarks = draw_landmarks(color_image_with_landmarks, mp_result.multi_hand_landmarks)

			# color_image_with_landmarks = cv2.rectangle(color_image_with_landmarks, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255))
			shm_mediapipe['color_image_with_landmarks'] = color_image_with_landmarks


			shm_mediapipe['color_image'] = color_image
			shm_mediapipe['depth_image'] = depth_image
			shm_mediapipe['depth_valid_area'] = depth_valid_area
			shm_mediapipe['frame_no'] = frame_no

			shm_mediapipe['fps'] = fps_counter.count()

	shm_flags['mediapipe_task_closed'] = True
	print("* MediaPipe Task Closed")


###############################################################################
# FastSAM
###############################################################################
import fastsam
# fastSAM_model = fastsam.FastSAM('model_checkpoint/FastSAM-s.pt')
fastSAM_model = fastsam.FastSAM('model_checkpoint/FastSAM-x.pt')


def do_fastsam(img: np.ndarray, plot_to_result=False):
	# with time_keeper("FastSAM everything_results"):
	everything_results = fastSAM_model(img, device=device, retina_masks=True, imgsz=256, conf=0.1, iou=0.5)
	# everything_results = fastSAM_model(img, device=DEVICE, retina_masks=True, imgsz=384, conf=0.1, iou=0.5)

	# with time_keeper("FastSAM prompt_process"):
	# prompt_process = fastsam.FastSAMPrompt(img, everything_results, device=device)

	# Everything Prompt
	# with time_keeper("FastSAM anns"):
	# anns = prompt_process.everything_prompt()
	anns = everything_results[0].masks.data

	# Point prompt
	# points default [[0,0]] [[x1,y1],[x2,y2]]
	# point_label default [0] [1,0] 0:background, 1:foreground
	# anns = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

	# Box Prompt
	# w, h, _ = img.shape
	# anns = prompt_process.box_prompt(bboxes=[[5, 5, w-5, h-5]]) # [x1, y1, x2, y2]

	# Text Prompt
	# anns = prompt_process.text_prompt(text='hand')

	# with time_keeper("FastSAM plot_to_result"):
	# global fastsam_visualization
	if plot_to_result:
		prompt_process = fastsam.FastSAMPrompt(img, everything_results, device=device)
		plot = prompt_process.plot_to_result(annotations=anns, mask_random_color=True)
	# cv2.imshow("FastSam Visualization", fastsam_visualization)

	masks = [ann.cpu().numpy() == 1 for ann in anns]

	if plot_to_result:
		return masks, plot
	else:
		return masks

def fastsam_subprocess_init():
	do_fastsam(zeros_color)


ave_count = 0
ave_max_count = 10
ave_sum = 0
ave = 0
last_submit = -1
def fastsam_task_scheduler(shm_mediapipe, shm_sa, shm_flags):
	# from FastSAM import fastsam
	# # fastSAM_model = fastsam.FastSAM('FastSAM/weights/FastSAM-s.pt')
	# fastSAM_model = fastsam.FastSAM('FastSAM/weights/FastSAM-x.pt')

	# with Manager() as manager:
	with DummyContext():
		# shared_fps = manager.dict()
		# shared_fps['count'] = 0
		# shared_fps['fps'] = 0
		# shared_fps['start_time'] = time.time()

		max_workers = 3
		# with ThreadPoolExecutor(max_workers=max_workers, initializer=fastsam_subprocess_init) as pool:
		with ProcessPoolExecutor(max_workers=max_workers, initializer=fastsam_subprocess_init) as pool:

			# Initialize process
			def do_nothing():
				pass
			for _ in range(max_workers):
				pool.submit(do_nothing)

			# Wait for MediaPipe
			print("Waiting for MediaPipe...")
			while not shm_mediapipe['ready']:
				time.sleep(0.01)
				if shm_flags['end_flag']:
					break

			print("Starting FastSAM task")

			fps_counter = Fps_Counter()

			def submit_subtask():
				if shm_flags['end_flag']:
					return
				future = pool.submit(fastsam_task, shm_mediapipe, shm_sa, shm_flags)
				future.add_done_callback(callback)

			def callback(future, min_delay=input_seconds_per_frame):
				if shm_flags['end_flag']:
					return

				global ave_count, ave_max_count, ave_sum, ave, last_submit

				ret = future.result()
				if ret > 0:
					# Calc average turn-around time
					callback.fail_count = 0
					fps = fps_counter.count()
					shm_sa['fps'] = fps
					ave_count += 1
					ave_sum += ret
					if ave_count >= ave_max_count:
						ave = ave_sum / ave_count
						ave_sum = 0
						ave_count = 0
						# print(f"ave: {ave*1000:.2f} ms")
				else: # task failed
					callback.fail_count += 1
					if callback.fail_count >= 3:
						shm_flags['reset'] = True

				if pool._queue_count >= max_workers:
					delay = ave / max_workers
					time.sleep(max(delay, min_delay))
					# t1 = time.time()
					# while time.time() - t1 < delay:
					# 	time.sleep(0.01)

				# Wait for a new frame
				# frame_no_prev = callback.frame_no
				# callback.frame_no = shm_mediapipe['frame_no']
				# while not (callback.frame_no > frame_no_prev):
				# 	if shm_flags['end_flag']:
				# 		break
				# 	callback.frame_no = shm_mediapipe['frame_no']

				t1 = time.time()
				if t1 - last_submit < min_delay:
					time.sleep(min_delay - (t1 - last_submit))

				last_submit = time.time()

				submit_subtask()
			callback.frame_no = 0
			callback.fail_count = 0

			for _ in range(max_workers):
				submit_subtask()
				time.sleep(input_seconds_per_frame)

			while not shm_flags['end_flag']:
				time.sleep(0.1)


	shm_flags['fastsam_task_closed'] = True
	print("* FastSAM Task Closed")


mask_hand_presition_thresh = 0.95
mask_hand_iou_thresh = 0.5

indices_matrix = np.indices((h, w))

def fastsam_task(shm_mediapipe, shm_sa, shm_flags):
	task_start_time = time.time()

	error_message = ''

	if shm_flags['end_flag']:
		return

	reset_flag = shm_flags['reset']

	color_image = shm_mediapipe['color_image']
	depth_image = shm_mediapipe['depth_image']
	frame_no = shm_mediapipe['frame_no']

	hand_bbox_size = shm_mediapipe['hand_bbox_size']
	depth_valid_area = shm_mediapipe['depth_valid_area']

	def error(message="", reset=False):
		shm_sa['color_image'] = color_image
		shm_sa['depth_image'] = depth_image
		shm_sa['depth_valid_area'] = depth_valid_area
		shm_sa['color_image_with_mask'] = color_image
		shm_sa['color_image_with_mask_hand'] = color_image
		shm_sa['error_message'] = message

		if reset:
			shm_flags['reset'] = True

		return -1


	mask_hand_prev_retrieved = shm_sa['mask_hand_retrieved']
	mask_hand_prev = shm_sa['mask_hand']


	##### Get hand bounding box #####
	"""
	# using MediaPipe
	if mask_hand_prev is None or reset_flag:
	# if True:
		hand_center = shm_mediapipe['hand_center']

		# Determine crop size
		if depth_valid_area[hand_center[1], hand_center[0]]:
			hand_depth = depth_image[hand_center[1], hand_center[0]]
			if hand_depth != 0:
				hand_bbox_size *= np.arctan(hand_bbox_size / hand_depth)
			else:
				hand_bbox_size *= np.arctan(hand_bbox_size)
		else:
			return error(message='Depth value out of range', reset=True)

		x1, y1 = hand_center - np.array([0.5, 0.5]) * hand_bbox_size
		x2, y2 = hand_center + np.array([0.5, 0.5]) * hand_bbox_size

		shm_flags['reset'] = False
		hand_bbox_color = (0, 0, 255)

	# using prev frame
	else:
		# Determine crop size
		depth_image_prev = shm_sa['depth_image']
		depth_valid_area_prev = shm_sa['depth_valid_area']
		if depth_valid_area_prev[mask_hand_prev].any():
			hand_depth = np.mean(depth_image_prev[depth_valid_area_prev & mask_hand_prev])
			if hand_depth != 0:
				hand_bbox_size *= np.arctan(hand_bbox_size / hand_depth)
			else:
				hand_bbox_size *= np.arctan(hand_bbox_size)
		else:
			return error(message='Depth value out of range', reset=True)

		indices_x = indices_matrix[1][mask_hand_prev]
		indices_y = indices_matrix[0][mask_hand_prev]
		x_min = np.min(indices_x)
		y_min = np.min(indices_y)
		x_max = np.max(indices_x)
		y_max = np.max(indices_y)

		hand_bbox_prev = shm_sa['hand_bbox'] # x1, y1, x2, y2
		margins_prev = np.array([x_min, y_min, x_max, y_max]) - np.array(hand_bbox_prev)
		margins_prev[2:4] = -margins_prev[2:4]

		margin = hand_bbox_size * 0.2
		margin_thresh = hand_bbox_size * 0.05

		margins_prev_large_enough = margins_prev > margin_thresh

		if margins_prev_large_enough[0] and margins_prev_large_enough[2]: # left-right
			center = (x_max + x_min) // 2
			x1 = center - hand_bbox_size // 2
			x2 = x1 + hand_bbox_size
		elif margins_prev_large_enough[0]: # left
			x1 = x_min - margin
			x2 = x1 + hand_bbox_size
		elif margins_prev_large_enough[2]: # right
			x2 = x_max + margin
			x1 = x2 - hand_bbox_size
		else:
			return error(message='No enough margin (1)', reset=True)

		if margins_prev_large_enough[1] and margins_prev_large_enough[3]: # top-bottom
			center = (y_max + y_min) // 2
			y1 = center - hand_bbox_size // 2
			y2 = y1 + hand_bbox_size
		elif margins_prev_large_enough[1]: # top
			y1 = y_min - margin
			y2 = y1 + hand_bbox_size
		elif margins_prev_large_enough[3]: # bottom
			y2 = y_max + margin
			y1 = y2 - hand_bbox_size
		else:
			return error(message='No enough margin (2)', reset=True)

		# hand_center = (x, y)
		# hand_center = image_center(mask_hand_prev_retrieved)
		# hand_center = shm_sa['hand_center'] - np.array(shm_sa['diff']) * 2
		# hand_center = (x2 - x1) // 2, (y2 - y1) // 2

		# hand_center = (int(hand_center[0]), int(hand_center[1]))

		hand_bbox_color = (255, 0, 0)

	shm_sa['hand_depth'] = hand_depth

	x1 = int(x1)
	y1 = int(y1)
	x2 = int(x2)
	y2 = int(y2)

	x1 = max(x1, 0)
	y1 = max(y1, 0)
	x2 = min(x2, w-1)
	y2 = min(y2, h-1)

	color_image_crop = color_image[y1:y2, x1:x2, :]
	shm_sa['hand_bbox'] = (x1, y1, x2, y2)
	shm_sa['hand_bbox_size'] = hand_bbox_size
	"""

	# Using Track-Anything
	mask_hand = shm_mediapipe['mask_hand']
	indices_x = indices_matrix[1][mask_hand]
	indices_y = indices_matrix[0][mask_hand]
	x_min = np.min(indices_x)
	y_min = np.min(indices_y)
	x_max = np.max(indices_x)
	y_max = np.max(indices_y)

	half_hand_bbox_size = int((max(x_max - x_min, y_max - y_min) * 1.4) / 2)
	hand_center = ((x_max + x_min)/2, (y_max + y_min)/2)

	x1 = hand_center[0] - half_hand_bbox_size
	x2 = hand_center[0] + half_hand_bbox_size
	y1 = hand_center[1] - half_hand_bbox_size
	y2 = hand_center[1] + half_hand_bbox_size

	x1 = int(x1)
	y1 = int(y1)
	x2 = int(x2)
	y2 = int(y2)

	x1 = max(x1, 0)
	y1 = max(y1, 0)
	x2 = min(x2, w-1)
	y2 = min(y2, h-1)

	hand_bbox_color = [0, 255, 0]

	color_image_crop = color_image[y1:y2, x1:x2, :]
	##### Get hand bounding box #####


	##### Do FastSAM #####
	try:
		everything_masks = do_fastsam(color_image_crop, plot_to_result=False)
		# everything_masks, v = do_fastsam(color_image_crop, plot_to_result=True)
		# _v = color_image.copy()
		# _v[y1:y2, x1:x2, :] = v
	except:
		return error(message='FastSAM failed', reset=True)

	# Revert cropping
	for i, mask in enumerate(everything_masks):
		tmp = zeros_bool.copy()
		tmp[y1:y2, x1:x2] = mask
		everything_masks[i] = tmp

	##### Do FastSAM #####


	##### Get mask_hand #####
	"""
	mask_hand_cache = shm_sa['mask_hand_cache']
	mask_hand_retrieved_cache = shm_sa['mask_hand_retrieved_cache']


	# using MediaPipe
	if (not mask_hand_cache.any()) or reset_flag:
		landmark_coords = shm_mediapipe['coords']
		presitions = [calc_landmark_presition(mask, landmark_coords) for mask in everything_masks]
		presitions_argmax = argmax(presitions)
		mask_hand = everything_masks[presitions_argmax]

	# using data from prev frame
	else:
		# Split distinct masks
		_everything_masks = []
		for mask in everything_masks:
			mask = bool2uint8(mask)
			num_labels, label_matrix = cv2.connectedComponents(mask)

			for i in range(num_labels-1):
				mask = (label_matrix == i+1) # [0] is background
				mask = binarize(mask)
				_everything_masks.append(mask)

		everything_masks = _everything_masks

		# Generate a mask that consists of any masks contained by mask_hand_prev
		mask_hand = zeros_bool.copy()
		mask_occluder_prev = shm_sa['mask_occluder']

		# Find a mask similar to prev occluder
		ious = [calc_iou(mask, mask_occluder_prev) for mask in everything_masks]
		iou_argmax = argmax(ious)
		if ious[iou_argmax] > 0.5:
			mask_occluder = everything_masks[iou_argmax]
		else:
			mask_occluder = zeros_bool.copy()

		for mask in everything_masks:
			# ignore masks similar to prev occluder
			mask = mask & (~mask_occluder)

			# Filter-out objects behind hand
			mask_depth = calc_mask_depth(mask, depth_image, depth_valid_area)
			if (mask_depth >= 0) and (mask_depth >= hand_depth * 1.2):
				continue

			for cached_mask_hand in mask_hand_cache.get_all():
			# for cached_mask_hand in mask_hand_retrieved_cache.get_all():
				contained_ratio = (cached_mask_hand & mask).sum() / mask.sum() if mask.any() else 0
				if contained_ratio > 0.5:
					mask_hand = mask_hand | mask
					break
		if mask_hand.any():
			everything_masks.append(mask_hand)
			# pass
		if True:
			pass
		else:
			ious = [calc_iou_fixed(mask, mask_hand_prev) for mask in everything_masks]
			iou_argmax = argmax(ious)

			if ious[iou_argmax] >= mask_hand_iou_thresh:
				mask_hand = everything_masks[iou_argmax]
			# elif presitions[presitions_argmax] >= mask_hand_presition_thresh:
			# 	mask_hand = fastsam_result[presitions_argmax]
			else:
				return error(message='Could not find mask_hand', reset=True)
				mask_hand = mask_hand_prev

		# # Remove distinct object
		# mask_hand = bool2uint8(mask_hand)
		# mask_hand = ndarray_gray2rgb(mask_hand)

		# num_labels, labels = cv2.connectedComponents(mask_hand)
		# mask_hand_segments = []
		# ious = []
		# for i in range(num_labels):
		# 	_mask = labels == i
		# 	mask_hand_segments.append(_mask)

		# 	ious.append(calc_iou(mask_hand_prev, _mask))

		# mask_hand = mask_hand_segments[argmax(ious)]

		# mask_hand = ndarray_rgb2gray(mask_hand)
		# mask_hand = binarize(mask_hand)


	# Filter by color
	# color_image_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
	# hand_h_median = np.median(color_image_hsv[mask_hand, 2])
	# print(hand_h_median)
	# mask_hand[abs(color_image_hsv[:, :, 2] - hand_h_median) > 20] = False

	shm_sa['mask_hand'] = mask_hand
	mask_hand_cache.put(mask_hand)
	shm_sa['mask_hand_cache'] = mask_hand_cache
	"""
	shm_sa['mask_hand'] = mask_hand

	# Remove mask_hand from everything_masks
	ious = [calc_iou(mask, mask_hand) for mask in everything_masks]
	iou_argmax = argmax(ious)
	if ious[iou_argmax] > 0.5:
		del everything_masks[iou_argmax]
	##### Get mask_hand #####


	##### Get occluder mask #####
	# Get depth mask
	"""
	depth_image_hand = depth_image * mask_hand
	depth_thresh = depth_image_hand.sum() / (mask_hand.sum()) # Average depth in mask_hand
	# depth_threshold += 30
	depth_mask = (depth_image <= depth_thresh)


	masks_in_front_of_hand = []
	mask = depth_mask & (~mask_hand)
	# _mask = depth_mask
	for mask_occluder in everything_masks:
		if calc_iou(mask_occluder, mask & mask_occluder) >= 0.4:
			masks_in_front_of_hand.append(mask_occluder & (~mask_hand))
			# masks.append(mask & _mask)
	"""
	masks_in_front_of_hand = []
	stat = mask_depth_stat(mask_hand, depth_image, depth_valid_area)
	if stat is not None:
		mask_hand_depth_min, mask_hand_depth_max, mask_hand_depth_median, mask_hand_depth_std = stat
		error_message = f'min {mask_hand_depth_min:3.2f}, max {mask_hand_depth_max:3.2f},\nmed {mask_hand_depth_median:3.2f}, std {mask_hand_depth_std:3.2f}'
		for mask in everything_masks:
			# if calc_iou(mask, mask_hand) > 0.8:
			# 	continue
			stat = mask_depth_stat(mask, depth_image, depth_valid_area)
			if stat is None:
				continue
			mask_depth_min, mask_depth_max, mask_depth_median, mask_depth_std = stat
			# if mask_depth_max <= mask_hand_depth_max:
			if mask_depth_max <= mask_hand_depth_max-mask_hand_depth_std and mask_depth_std < 30:
				masks_in_front_of_hand.append(mask)
	else:
		mask_hand_depth_min, mask_hand_depth_max, mask_hand_depth_median, mask_hand_depth_std = None, None, None, None
		error_message = 'No depth in mask_hand is valid'

	# Compose all masks into one image
	mask_in_front_of_hand = zeros_bool.copy()
	for mask in masks_in_front_of_hand:
		mask_in_front_of_hand = mask_in_front_of_hand | mask

	mask_occluder = zeros_bool
	mask_hand_retrieved = zeros_bool

	"""
	# Retrieve mask_hand
	if (mask_hand is not None) and (mask_hand_prev is not None):
		lack_prev = shm_sa['lack']

		mask_hand_fixed, mask_hand_prev_fixed, diff1, diff2 = fix_mask(mask_hand, mask_hand_prev)
		lack_prev = translate(lack_prev, diff2[0], diff2[1])
		mask_hand_prev_fixed = mask_hand_prev_fixed | lack_prev

		lack = mask_hand_prev_fixed ^ mask_hand_fixed
		lack = translate(lack, -diff1[0], -diff1[1])

		# lack = (mask_hand_prev | lack) & (~mask_hand)
		lack = lack & mask_in_front_of_hand

		# mask_prev = shm_sa['mask']
		# if mask_prev is not None:
		# 	_lack = (~mask_hand_prev) & mask_hand
		# 	_lack = _lack & mask_prev
		# 	lack = lack | _lack

		lack = bool2uint8(lack)
		lack = cv2.morphologyEx(lack, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8), iterations = 3)
		lack = binarize(lack, threshold=1)

		mask_hand_retrieved = mask_hand | lack
		# mask_hand = binarize(cv2.morphologyEx(ndarray_bool2uint8(mask_hand), cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8), iterations = 3), threshold=1)

		mask_occluder = lack

		shm_sa['lack'] = lack
	else:
		lack = zeros_bool

	# Find occluder object
	mask_occluder_object = zeros_bool.copy()
	if mask_occluder.any():
		for mask in masks_in_front_of_hand:
			# mask = mask & (~mask_hand)
			# ratio =  (mask & mask_occluder).sum() / mask_occluder.sum()
			# if ratio > 0.25:
			m = mask & mask_occluder

			if m.any():
				mask_occluder_object = mask_occluder_object | mask

			# if not m.any():
			# 	continue
			# stat = mask_depth_stat(m, depth_image, depth_valid_area)
			# if stat is None:
			# 	continue

			# m_min, m_max, m_median, m_std = stat
			# if (m_median < mask_hand_depth_median):
			# 	mask_occluder_object = mask_occluder_object | mask

	mask_occluder = mask_occluder_object
	"""

	mask_occluder_object = zeros_bool.copy()
	mask_hand_edge = mask_edge(mask_hand, thickness=10)
	for mask in masks_in_front_of_hand:
		m = mask & mask_hand_edge
		if not m.any():
			continue

		stat = mask_depth_stat(m, depth_image, depth_valid_area, erosion_kernel_size=3)
		if stat is None:
			continue

		m_depth_min, m_depth_max, m_depth_median, m_depth_std = stat
		if m_depth_min < mask_hand_depth_max:
			mask_occluder_object = mask_occluder_object | mask

	mask_occluder = mask_occluder_object


	shm_sa['mask_occluder'] = mask_occluder
	shm_sa['mask_hand_retrieved'] = mask_hand_retrieved
	# mask_hand_retrieved_cache.put(mask_hand_retrieved)
	# shm_sa['mask_hand_retrieved_cache'] = mask_hand_retrieved_cache

	color_image_with_mask = add_mask(color_image, mask_occluder)
	##### Get occluder mask #####


	color_image_with_mask_hand = add_mask(color_image, mask_hand)
	cv2.rectangle(color_image_with_mask_hand, pt1=(x1, y1), pt2=(x2, y2), color=hand_bbox_color)

	shm_sa['color_image_with_mask_hand'] = color_image_with_mask_hand
	shm_sa['color_image_with_mask'] = color_image_with_mask
	shm_sa['color_image'] = color_image
	shm_sa['depth_image'] = depth_image
	shm_sa['depth_valid_area'] = depth_valid_area
	shm_sa['frame_no'] = frame_no

	shm_sa['error_message'] = error_message


	return time.time() - task_start_time



###############################################################################
# Main
###############################################################################
if __name__ == "__main__" :
	with ThreadPoolExecutor(max_workers=3, initializer=init_process) as pool:
	# with ProcessPoolExecutor(max_workers=3, initializer=init_process) as pool:
	# with Pool(processes=5, initializer=init_process) as pool:
		with Manager() as manager:
			# Flags
			shm_flags = manager.dict()
			shm_flags['end_flag'] = False
			shm_flags['rgbd_streaming_task_closed'] = False
			shm_flags['mediapipe_task_closed'] = False
			shm_flags['fastsam_task_closed'] = False
			shm_flags['reset'] = False

			# RGBD Streaming
			shm_rgbd = manager.dict()
			shm_rgbd['color_image'] = zeros_color
			shm_rgbd['depth_image'] = zeros_gray
			shm_rgbd['depth_valid_area'] = None
			shm_rgbd['frame_no'] = 0
			shm_rgbd['fps'] = 0

			# MediaPipe
			shm_mediapipe = manager.dict()
			shm_mediapipe['ready'] = False
			shm_mediapipe['coords'] = None
			shm_mediapipe['hand_bbox_size'] = 0
			shm_mediapipe['hand_bbox_size_adjust_count'] = 0
			shm_mediapipe['hand_bbox'] = (0, 0, w-1, h-1)
			shm_mediapipe['hand_center'] = (0, 0)
			shm_mediapipe['mask_hand'] = zeros_bool
			shm_mediapipe['color_image'] = zeros_color
			shm_mediapipe['depth_image'] = zeros_gray
			shm_mediapipe['depth_valid_area'] = zeros_bool
			shm_mediapipe['color_image_with_landmarks'] = zeros_color
			shm_mediapipe['frame_no'] = 0
			shm_mediapipe['fps'] = 0

			# Segment Anything
			shm_sa = manager.dict()
			shm_sa['color_image'] = zeros_color
			shm_sa['depth_image'] = zeros_gray
			shm_sa['depth_valid_area'] = zeros_bool
			shm_sa['mask_occluder'] = zeros_bool
			shm_sa['mask_hand'] = None
			# shm_sa['mask_hand_cache'] = Cache(max_size=5)
			shm_sa['mask_hand_retrieved'] = None
			# shm_sa['mask_hand_retrieved_cache'] = Cache(max_size=5)
			shm_sa['hand_depth'] = 0
			shm_sa['color_image_with_mask'] = zeros_bool
			shm_sa['lack'] = zeros_bool
			shm_sa['color_image_with_mask_hand'] = zeros_color
			shm_sa['hand_bbox'] = (0, 0, w-1, h-1)
			shm_sa['hand_bbox_size'] = 0
			shm_sa['test'] = None
			shm_sa['frame_no'] = 0
			shm_sa['fps'] = 0
			shm_sa['error_message'] = ''

			output_dir = 'output'
			output_filename = time.strftime('%Y-%m%d-%H%M%S')
			output_filepath = os.path.join(output_dir, output_filename)
			os.makedirs(output_dir, exist_ok=True)

			if record_in_video_cv2:
				writer = cv2.VideoWriter(
					output_filepath,
					cv2.VideoWriter_fourcc(*'mp4v'),
					fps=30,
					frameSize=(w*2, h),
					isColor=True
				)
				output_filename += '.mp4'
				def write(color_image, mask_image):
					writer.write(np.hstack([
						color_image,
						gray2rgb(mask_image)
					]))

				def close_writer():
					writer.release()

			elif record_in_video_ffmpeg:
				process = (
					ffmpeg
					.input('pipe:', format='rawvideo', pix_fmt='bgr24',
						s=f'{w*2}x{h}', use_wallclock_as_timestamps=1)
					.output(output_filepath, vsync='vfr', r=input_fps)
					.overwrite_output()
					.run_async(pipe_stdin=True)
				)

				output_filename += '.mp4'

				def write(color_image, mask_image):
					process.stdin.write(np.hstack([
						color_image,
						gray2rgb(mask_image)
					]).tobytes())

				def close_writer():
					process.stdin.close()
					process.wait()

			elif record_in_images:
				output_dir = os.path.join(output_dir, output_filename)
				output_dir_rgb = os.path.join(output_dir, 'rgb')
				output_dir_mask = os.path.join(output_dir, 'mask')
				os.makedirs(output_dir_rgb, exist_ok=True)
				os.makedirs(output_dir_mask, exist_ok=True)

				def write(color_image, mask_image):
					write.n += 1
					filename = f'{write.n:0>5}.png'
					output_filename_rgb = os.path.join(output_dir_rgb, filename)
					output_filename_mask = os.path.join(output_dir_mask, filename)
					cv2.imwrite(output_filename_rgb, color_image)
					cv2.imwrite(output_filename_mask, bool2uint8(mask_image))
				write.n = 0

				def close_writer():
					pass

			else:
				def write(_, __):
					pass

				def close_writer():
					pass


			# Start processes
			pool.submit(rgbd_streaming_task, shm_rgbd, shm_flags)
			pool.submit(mediapipe_task, shm_rgbd, shm_mediapipe, shm_flags)
			pool.submit(fastsam_task_scheduler, shm_mediapipe, shm_sa, shm_flags)

			fps_counter = Fps_Counter()
			# Show result
			# while True:
			frame_no = -1
			while not shm_flags['end_flag']:
				color_image = shm_rgbd['color_image']
				depth_image = shm_rgbd['depth_image']
				color_image_with_landmarks = shm_mediapipe['color_image_with_landmarks']

				# color_image2 = shm_mediapipe['color_image']
				# mask_hand = shm_mediapipe['mask_hand']
				# color_image2 = add_mask(color_image2, mask_hand)

				color_image3 = shm_sa['color_image']
				color_image_with_mask = shm_sa['color_image_with_mask']
				masked_hand_image = shm_sa['color_image_with_mask_hand']
				mask_occluder = shm_sa['mask_occluder']
				lack = shm_sa['lack']
				test = shm_sa['test']
				hand_bbox_size = shm_sa['hand_bbox_size']

				frame_no_prev = frame_no
				frame_no = shm_sa['frame_no']

				error_message = shm_sa['error_message']

				# if frame_no > frame_no_prev:
				# 	fps_counter.count()


				info_image = zeros_color.copy()
				cv2.putText(info_image, f"Input FPS: {shm_rgbd['fps']:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), thickness=2)
				cv2.putText(info_image, f"Output FPS: {shm_sa['fps']:.2f}", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), thickness=2)


				cv2.putText(info_image, 'Info', (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), thickness=2)
				if error_message:
					row = 150
					cv2.putText(info_image, 'Error:', (40, row), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), thickness=2)
					for line in error_message.split('\n'):
						cv2.putText(info_image, line, (150, row), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), thickness=2)
						row += 30
				# cv2.putText(info_image, 'bbox:' + str(hand_bbox_size), (40, 180), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), thickness=2)
				# cv2.putText(info_image, str(shm_sa['fps']), (40, 210), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), thickness=2)

				image = xstack([
					color_image
					, depth_image
					# , color_image_with_landmarks
					, masked_hand_image
					, color_image_with_mask
					, info_image
					# , test
					# , color_image2
				])

				if frame_no > frame_no_prev:
					write(color_image3, mask_occluder)

				cv2.imshow("", image)

				key = cv2.waitKey(1)
				if key == 27: # Esc
					shm_flags['end_flag'] = True
					cv2.destroyAllWindows()
					break
				elif key == ord('r'):
					shm_flags['reset'] = True
					shm_mediapipe['hand_bbox_size_adjust_count'] = 0

				if (
					shm_flags['rgbd_streaming_task_closed']
					or shm_flags['mediapipe_task_closed']
					or shm_flags['fastsam_task_closed']
				):
					shm_flags['end_flag'] = True
					break

				time.sleep(0.005)

			# Exit program
			close_writer()

			while not (
				shm_flags['rgbd_streaming_task_closed']
				and shm_flags['mediapipe_task_closed']
				and shm_flags['fastsam_task_closed']
			):
				time.sleep(0.1)

	cv2.destroyAllWindows()
	print("*** program exited ***")
