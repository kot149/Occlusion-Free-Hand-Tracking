import subprocess
from tkinter import filedialog
import os
import time
from termcolor import colored, cprint

def run_command(command):
	if not command:
		print('Null command')
		return

	command = [str(c) for c in command]

	cprint(f'Runing command {command}', 'green')
	return subprocess.run(command, shell=True)


output_path = 'test' + time.strftime('%Y-%m%d-%H%M%S')

#########################################################
# Run Auto Mask Generator
#########################################################
run_command([
	'conda', 'activate', 'ofht', '&&'
	, 'python', 'src/ofht.py'
	, '-s' # Save result
	# , '-i', 'record/ex3_p2_1.mp4' # Input file
	, '-n', '7' # max_workers
	, '-o', output_path # Output folder
	, '-e' # Exit on record stop
])

#########################################################
# Run E2FGVI
#########################################################
input_dir = os.path.join(r'..\Occlusion-Free-Hand-Tracking\output', output_path)
video = os.path.join(input_dir, 'rgb.mp4')
if not os.path.exists(video):
	video = os.path.join(input_dir, 'rgb')
mask = os.path.join(input_dir, 'mask')
run_command([
	'cd', r'..\E2FGVI', '&&',
	'conda', 'activate', 'ofht2', '&&'
	, 'python', 'test_mod.py'
	, '--model', 'e2fgvi_hq'
	, '--ckpt', 'release_model/E2FGVI-HQ-CVPR22.pth'
	, '--video', video
	, '--mask', mask
	, '--num_ref', 10
	, '--step', 20
	, '--neighbor_stride', 3
	, '--savefps', 29
	, '--set_size', '--width', 640, '--height', 360
])

#########################################################
# Run failure_count
#########################################################
run_command([
	'conda', 'activate', 'ofht', '&&'
	, 'python', 'src/failure_count.py'
	, '-s' # Save Result
	, '-i', 'output/' + output_path # Input file
	, '-o', 'test.mp4' # Output file
])