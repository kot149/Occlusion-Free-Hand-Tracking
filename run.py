import subprocess
from tkinter import filedialog
import os

def run_command(command):
	if not command:
		print('Null command')
		return

	command = [str(c) for c in command]

	print(f'Runing command \'{command}\'')
	return subprocess.run(command, shell=True)

run_command([
	'conda', 'activate', 'ofht', '&&'
	, 'python', 'src/failure_count.py'
	, '-s'
	, '-i', 'output/ex3_np2_1_p'
	, '-o', 'test.mp4'
])