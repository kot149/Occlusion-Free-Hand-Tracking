
import datetime
import numpy as np
import torch
import cv2

import fastsam
# fastSAM_model = fastsam.FastSAM('model_checkpoint/FastSAM-s.pt')
fastSAM_model = fastsam.FastSAM('model_checkpoint/FastSAM-x.pt')

DEVICE = torch.device("cuda")
def do_fastsam(img: np.ndarray, points, plot_to_result=False):
	# with time_keeper("FastSAM everything_results"):
	everything_results = fastSAM_model(img, device=DEVICE, retina_masks=True, imgsz=256, conf=0.1, iou=0.5)
	# everything_results = fastSAM_model(img, device=DEVICE, retina_masks=True, imgsz=384, conf=0.1, iou=0.5)

	# with time_keeper("FastSAM prompt_process"):
	prompt_process = fastsam.FastSAMPrompt(img, everything_results, device=DEVICE)

	# Everything Prompt
	# with time_keeper("FastSAM anns"):
	# anns = prompt_process.everything_prompt()
	# anns = everything_results[0].masks.data

	# Point prompt
	# points default [[0,0]] [[x1,y1],[x2,y2]]
	# point_label default [0] [1,0] 0:background, 1:foreground
	anns = prompt_process.point_prompt(points=points, pointlabel=[1])

	# Box Prompt
	# w, h, _ = img.shape
	# anns = prompt_process.box_prompt(bboxes=[[5, 5, w-5, h-5]]) # [x1, y1, x2, y2]

	# Text Prompt
	# anns = prompt_process.text_prompt(text='hand')

	# with time_keeper("FastSAM plot_to_result"):
	# global fastsam_visualization
	if plot_to_result:
		plot = prompt_process.plot_to_result(annotations=anns, mask_random_color=True)
	# cv2.imshow("FastSam Visualization", fastsam_visualization)

	if plot_to_result:
		return [ann.cpu().numpy() for ann in anns], plot
	else:
		# return [ann.cpu().numpy() for ann in anns]
		return anns

import importlib
# track_anything = importlib.import_module("Track-Anything")
from track_anything.tracker.base_tracker import BaseTracker

#BaseTrackerの初期化
xmem_checkpoint = 'model_checkpoint/XMem-s012.pth'
device = "cuda"
tracker = BaseTracker(xmem_checkpoint, device)

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())


# Videoの初期化
cap = cv2.VideoCapture(3)
# cap = cv2.VideoCapture("C:\Users\Ku0143\GoogleDrive_k\KIT\lab\hand-tracking\Occlusion-Free-Hand-Tracking\record\2023-1218-193048.mp4")
if cap.isOpened() is False:
	raise IOError
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
_, frame = cap.read()
h, w, _ = frame.shape


#　各種変数定義
input_point = None
input_label = np.array([1])
clicked_frame = None
target_name = ""

# マウスイベントの処理関数を定義
def mouse_event(event, x, y, flags, param):
	global input_point, target_name, clicked_frame
	if event == cv2.EVENT_LBUTTONDOWN:
		input_point = np.array([[x, y]])
		# target_name = input("Enter a name for this point: ")
		target_name = 'hand'
		clicked_frame = frame.copy()

# マウスイベント時に処理を行うウィンドウの名前を設定
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_event)

print("start.")

# 最初のフレームを表示し、任意の点をクリックするまで待つ
while input_point is None:
	ret, frame = cap.read()
	if ret is False:
		raise IOError
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)

first_frame = True

# セグメンテーション
masks = do_fastsam(clicked_frame, input_point)
print(masks.shape)

# トラッキング
mask, prob, painted_frame = tracker.track(clicked_frame, masks[0])
first_frame = False

while True:
	try:
		ret, frame = cap.read()
		if ret is False:
			raise IOError

		# elseだけでいい
		if first_frame:
			mask, prob, painted_frame = tracker.track(clicked_frame, masks[0])
			first_frame = False
		else:
			mask, prob, painted_frame = tracker.track(frame)

		frame[mask > 0, :] = np.array([0, 0, 255])

		# true_points = np.where(mask)

		# 検出がなかったときように分岐
		# if true_points[0].size > 0 and true_points[1].size > 0:

			# top_left = true_points[1].min(), true_points[0].min()
			# bottom_right = true_points[1].max(), true_points[0].max()

			# color = (0, 0, 255)  # red
			# thickness = 2
			# cv2.rectangle(frame, top_left, bottom_right, color, thickness)

			# バウンディボックスやターゲット名をCVの画面上に描画![Something went wrong]()

			# text = target_name
			# org = (top_left[0], top_left[1] - 10)
			# font = cv2.FONT_HERSHEY_SIMPLEX
			# fontScale = 1
			# color = (255, 255, 255)  # white
			# cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


		cv2.imshow("Frame", frame)
		# cv2.imshow("Mask", mask * 255)
		key = cv2.waitKey(1)
		if key == 27:
			break


	except KeyboardInterrupt:
		break

cap.close()
cv2.destroyAllWindows()