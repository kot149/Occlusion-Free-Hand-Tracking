import numpy as np
import cv2

import joblib
from tkinter import filedialog
import os

w, h = 640, 360

def read_frames_from_images(input_dir: str):
	count = 1
	filepath_list = []
	filepath = os.path.join(input_dir, f'{count:0>5}.png')

	while os.path.exists(filepath):
		filepath_list.append(filepath)

		count += 1
		filepath = os.path.join(input_dir, f'{count:0>5}.png')

	frames = joblib.Parallel(n_jobs=-1)(joblib.delayed(cv2.imread)(f) for f in filepath_list)

	return frames, None


def read_frames_from_video(video_path: str):
	cap = cv2.VideoCapture(video_path)
	frames = []
	while cap.isOpened():
		success, frame = cap.read()
		if not success:
			break

		frames.append(frame)

	fps = cap.get(cv2.CAP_PROP_FPS)

	return frames, fps



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

if __name__ == '__main__':
	pole_image = cv2.imread(r'resource\pole.png')
	pole_color = pole_image[:, 0:w, :]
	pole_depth = pole_image[:, w:w*2, :]
	pole_depth_v = pole_image[:, w*2:w*3, :]

	pole_mask_color = binarize(cv2.imread(r'resource\pole_mask_color.png'))
	pole_mask_depth = binarize(cv2.imread(r'resource\pole_mask_depth.png'))

	input_dir = r'record'
	input_filepath = filedialog.askopenfilename(initialdir = input_dir)
	if not input_filepath:
		exit(-1)

	splitext = os.path.splitext(input_filepath)
	input_filepath_no_ext = splitext[0]
	input_filepath_ext = splitext[1]
	output_filepath = input_filepath_no_ext + '_p' + input_filepath_ext

	cap = cv2.VideoCapture(input_filepath)
	fps = cap.get(cv2.CAP_PROP_FPS)

	writer = cv2.VideoWriter(
		output_filepath,
		cv2.VideoWriter_fourcc(*'H264'),
		fps=fps,
		frameSize=(w*3, h),
		isColor=True
	)

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_color = frame[:, 0:w, :]
		frame_depth = frame[:, w:w*2, :]
		frame_depth_v = frame[:, w*2:w*3, :]

		frame_color[pole_mask_color] = pole_color[pole_mask_color]
		frame_depth[pole_mask_depth] = pole_depth[pole_mask_depth]
		frame_depth_v[pole_mask_depth] = pole_depth_v[pole_mask_depth]

		frame_with_pole = np.hstack([frame_color, frame_depth, frame_depth_v])

		writer.write(frame_with_pole)

		# cv2.imshow("", frame_with_pole)
		# if cv2.waitKey(5) == 27:
		# 	break

	cap.release()
	writer.release()
	cv2.destroyAllWindows()
	print(f'Result saved to "{output_filepath}"')