import sys, os
from util import *
from tkinter import filedialog
from tqdm import tqdm

if __name__ == '__main__':
	if len(sys.argv) > 1:
		input_dir = sys.argv[1]
	else:
		input_dir = filedialog.askdirectory(initialdir = 'output')

	print(f'input: {input_dir}')
	output_filepath = os.path.join(input_dir, 'rgb_mask.mp4')

	if os.path.exists(output_filepath):
		i = input(f"'{output_filepath} already exists.\nDo you want to continue? [y/n]\n")
		if i.lower() != 'y':
			print('Canceled.')
			exit(-1)

	frames_rgb = read_frames_from_images(os.path.join(input_dir, 'rgb'))
	frames_mask = read_frames_from_images(os.path.join(input_dir, 'mask'))


	writer = cv2.VideoWriter(
		output_filepath,
		cv2.VideoWriter_fourcc(*'mp4v'),
		fps=30,
		frameSize=(w*2, h),
		isColor=True
	)


	progress = tqdm(total=len(frames_rgb))
	for f, m in zip(frames_rgb, frames_mask):
		writer.write(np.hstack([f, m]))
		progress.update()

	writer.release()
	sys.stdout.flush()
	print('Done.')