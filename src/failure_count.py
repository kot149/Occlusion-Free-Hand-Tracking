import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import cv2
import numpy as np
import os
import time
import math
from tqdm import tqdm

import joblib
from tkinter import filedialog

def read_frames_from_images(input_dir: str):
	count = 1
	filepath_list = []
	filepath = os.path.join(input_dir, f'{count:0>5}.png')

	while os.path.exists(filepath):
		filepath_list.append(filepath)

		count += 1
		filepath = os.path.join(input_dir, f'{count:0>5}.png')

	frames = joblib.Parallel(n_jobs=-1)(joblib.delayed(cv2.imread)(f) for f in tqdm(filepath_list))

	return frames, None


def read_frames_from_video(video_path: str):
	cap = cv2.VideoCapture(video_path)
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	progress = tqdm(total=num_frames)
	frames = []
	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			break

		frames.append(frame)
		progress.update()

	fps = cap.get(cv2.CAP_PROP_FPS)

	return frames, fps

def add_mask(image_base: np.ndarray, mask: np.ndarray, color=(0, 0, 255)):
	result = image_base.copy()
	color = np.array(color)

	# mask_color = np.zeros_like(image_base)
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

def xstack(imgs : list):
	imgs = [(gray2rgb(img) if img is not None else np.zeros_like(img)) for img in imgs]
	n = len(imgs)
	num_cols = math.ceil(math.sqrt(n))
	num_rows = math.ceil(n / num_cols)

	for _ in range(num_cols * num_rows - n):
		imgs.append(np.zeros_like(imgs[0]))

	rows = []
	i = 0
	for r in range(num_rows):
		rows.append(np.hstack(imgs[i:i+num_cols]))
		i += num_cols

	return np.vstack(rows)

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

		# Draw multiple hands
		# for hand_landmarks in multi_hand_landmarks:
		# 	mp_drawing.draw_landmarks(
		# 		image,
		# 		hand_landmarks,
		# 		mp_hands.HAND_CONNECTIONS,
		# 		mp_drawing_styles.get_default_hand_landmarks_style(),
		# 		mp_drawing_styles.get_default_hand_connections_style()
		# 	)

	return image

if __name__ == '__main__':
	model_complexity = 0
	min_detection_confidence = 0.1
	min_tracking_confidence = 0.8

	input_dir = r'output'
	input_dir = filedialog.askdirectory(initialdir = input_dir)
	if not input_dir:
		exit(-1)
	print('Reading original RGB...')
	frames_rgb, fps = read_frames_from_images(os.path.join(input_dir, 'rgb'))
	print('Reading depth...')
	frames_mask, fps = read_frames_from_images(os.path.join(input_dir, 'mask'))
	print('Reading inpainted RGB...')
	frames_inpainted, fps = read_frames_from_video(os.path.join(input_dir, 'rgb_inpainted.mp4'))
	print('Preprocessing...')
	frames_mask = [binarize(rgb2gray(m)) for m in frames_mask]
	frames_rgb_masked = [add_mask(f, m) for f, m in zip(frames_rgb, frames_mask)]

	print('Done.')

	# if fps == None:
	if True:
		fps = 29


	writer = cv2.VideoWriter(
		os.path.join(input_dir, 'failure_count.mp4'),
		cv2.VideoWriter_fourcc(*'H264'),
		fps=fps,
		frameSize=(640*3, 360*2),
		isColor=True
	)

	zeros_color = np.zeros_like(frames_rgb[0])

	cv2.imshow('result', zeros_color)
	cv2.setWindowProperty('result', cv2.WND_PROP_TOPMOST, 1)

	crop_x1 = 0
	crop_x2 = 640
	crop_y1 = 0
	crop_y2 = 360

	frame_count = 0
	failure_count_orig = 0
	frame_count2 = 0
	failure_count_inapinted = 0

	with mp_hands.Hands(
		model_complexity=model_complexity,
		min_detection_confidence=min_detection_confidence,
		min_tracking_confidence=min_tracking_confidence,
		# num_hands=1
	) as hands_orig, mp_hands.Hands(
		model_complexity=model_complexity,
		min_detection_confidence=min_detection_confidence,
		min_tracking_confidence=min_tracking_confidence,
		# num_hands=1
	) as hands_inpainted:
		progress = tqdm(total=len(frames_rgb))
		for f, f_masked, f_inpainted in zip(frames_rgb, frames_rgb_masked, frames_inpainted):
			t = time.time()

			f_crop = f[crop_y1:crop_y2, crop_x1:crop_x2, :]
			f_inpainted_crop = f_inpainted[crop_y1:crop_y2, crop_x1:crop_x2, :]

			mp_result_orig = hands_orig.process(cv2.cvtColor(f_crop, cv2.COLOR_BGR2RGB))
			mp_result_inpainted = hands_inpainted.process(cv2.cvtColor(f_inpainted_crop, cv2.COLOR_BGR2RGB))

			frame_count += 1
			if mp_result_orig and mp_result_orig.multi_hand_landmarks:
				f_with_landmark = draw_landmarks(f_crop, mp_result_orig.multi_hand_landmarks)
				_f_with_landmark = f.copy()
				_f_with_landmark[crop_y1:crop_y2, crop_x1:crop_x2, :] = f_with_landmark
				f_with_landmark = _f_with_landmark
			else:
				failure_count_orig += 1
				f_with_landmark = f

			frame_count2 += 1
			if mp_result_inpainted and mp_result_inpainted.multi_hand_landmarks:
				f_inpainted_with_landmark = draw_landmarks(f_inpainted_crop, mp_result_inpainted.multi_hand_landmarks)
				_f_inpainted_with_landmark = f_inpainted.copy()
				_f_inpainted_with_landmark[crop_y1:crop_y2, crop_x1:crop_x2, :] = f_inpainted_with_landmark
				f_inpainted_with_landmark = _f_inpainted_with_landmark
			else:
				failure_count_inapinted += 1
				f_inpainted_with_landmark = f_inpainted

			info_message = f"""Parameters
    min_detection_confidence: {min_detection_confidence}
    min_tracking_confidence: {min_tracking_confidence}

Tracking failure count
    Original video:  {failure_count_orig:4}
    Inpainted video: {failure_count_inapinted:4}

Info
"""
			row = 30
			info_image = zeros_color.copy()

			for line in info_message.split('\n'):
				cv2.putText(info_image, line, (40, row), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), thickness=2)
				row += 30


			result = xstack([f, f_masked, f_inpainted, f_with_landmark, f_inpainted_with_landmark, info_image])
			writer.write(result)
			cv2.imshow('result', result)

			key = cv2.waitKey(1)
			if key == 27:
				break

			delay = 1/fps - (time.time() - t)
			# if delay > 0:
			# 	time.sleep(delay)

			progress.update()

	writer.release()
	cv2.destroyAllWindows()