from tkinter import filedialog
from icecream import ic
import numpy as np
import cv2
import sys, os
from tqdm import tqdm

input_filepath = r'record'
input_filepath = filedialog.askopenfilename(initialdir = input_filepath)
if not input_filepath:
	exit(-1)
print("Input file: ", input_filepath)
output_filename = os.path.splitext(os.path.basename((input_filepath)))[0]


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

frames = read_frames_from_video(input_filepath)

output_dir = 'output'
output_dir = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, 'rgb_no_occ.mp4')
