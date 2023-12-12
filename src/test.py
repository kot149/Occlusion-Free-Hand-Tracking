import numpy as np
from PIL import Image
import cv2
import torch

DEVICE = torch.device('cuda')

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



def e2fgvi_task(shm_sa, shm_e2fgvi, shm_flags):
	from PIL import Image
	import importlib
	from E2FGVI.core.utils import to_tensors


	# Load model
	net = importlib.import_module('.model.e2fgvi', package='E2FGVI')
	model = net.InpaintGenerator().to(DEVICE)
	data = torch.load('src/E2FGVI/release_model/E2FGVI-CVPR22.pth', map_location=DEVICE)
	model.load_state_dict(data)
	model.eval()

	frame_no = 0
	while not shm_flags['end_flag']:
		# Wait for a new frame
		frame_no_prev = frame_no
		frame_no = shm_sa['frame_no']
		# while not (frame_no > frame_no_prev):
		# 	if shm_flags['end_flag']:
		# 		break
		# 	frame_no = shm_sa['frame_no']

		# color_image = shm_sa['color_image']
		depth_image = shm_sa['depth_image']
		depth_valid_area = shm_sa['depth_valid_area']
		# mask_occluder = shm_sa['mask_occluder']
		color_image = cv2.imread("C:/Users/Takeuchi/Google Dirve/KIT/lab/hand-tracking/Occlusion-Free-Hand-Tracking/output/2023-1120-195227/rgb/00292.png")
		mask_occluder = cv2.imread("C:/Users/Takeuchi/Google Dirve/KIT/lab/hand-tracking/Occlusion-Free-Hand-Tracking/output/2023-1120-195227/mask/00292.png")
		mask_occluder = binarize(rgb2gray(mask_occluder))

		# Load as PIL Image
		color_image_pil = Image.fromarray(bgr2rgb(color_image))
		mask_occluder_pil = Image.fromarray(bool2uint8(mask_occluder))

		# Resize
		size_e2fgvi = (432, 240)
		color_image_pil = color_image_pil.resize(size_e2fgvi)
		mask_occluder_pil = mask_occluder_pil.resize(size_e2fgvi)

		color_image_pil_cache = shm_e2fgvi['color_image_pil_cache']
		mask_occluder_pil_cache = shm_e2fgvi['mask_occluder_pil_cache']

		color_image_pil_cache.put(color_image_pil)
		mask_occluder_pil_cache.put(mask_occluder_pil)

		shm_e2fgvi['color_image_pil_cache'] = color_image_pil_cache
		shm_e2fgvi['mask_occluder_pil_cache'] = mask_occluder_pil_cache


		# Prepare for inpainting
		frames = color_image_pil_cache.get_all()
		num_frames = len(frames)
		if num_frames < 5:
			continue
		imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
		frames = [np.array(f).astype(np.uint8) for f in frames]

		masks = mask_occluder_pil_cache.get_all()
		binary_masks = [
			np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
		]
		masks = to_tensors()(masks).unsqueeze(0)
		imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
		comp_frames = [None] * num_frames

		selected_imgs = imgs[:1, :, :, :, :]
		selected_masks = masks[:1, :, :, :, :]

		with torch.no_grad():
			masked_imgs = selected_imgs * (1 - selected_masks)
			mod_size_h = 60
			mod_size_w = 108
			h_pad = (mod_size_h - size_e2fgvi[1] % mod_size_h) % mod_size_h
			w_pad = (mod_size_w - size_e2fgvi[0] % mod_size_w) % mod_size_w
			masked_imgs = torch.cat(
				[masked_imgs, torch.flip(masked_imgs, [3])],
				3)[:, :, :, :size_e2fgvi[1] + h_pad, :]
			masked_imgs = torch.cat(
				[masked_imgs, torch.flip(masked_imgs, [4])],
				4)[:, :, :, :, :size_e2fgvi[0] + w_pad]
			pred_imgs, _ = model(masked_imgs, 1)
			pred_imgs = pred_imgs[:, :, :size_e2fgvi[1], :size_e2fgvi[0]]
			pred_imgs = (pred_imgs + 1) / 2
			pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
			# for i in range(size_e2fgvi):
			idx = num_frames-1
			i = idx
			img_inpainted = np.array(pred_imgs[i]).astype(
				np.uint8) * binary_masks[idx] + frames[idx] * (
					1 - binary_masks[idx])
			# if comp_frames[idx] is None:
			# 	comp_frames[idx] = img
			# else:
			# 	comp_frames[idx] = comp_frames[idx].astype(
			# 			np.float32) * 0.5 + img.astype(np.float32) * 0.5
			img_inpainted = bgr2rgb(img_inpainted.astype(np.uint8))

		shm_e2fgvi['color_image_inpainted'] = img_inpainted

		shm_e2fgvi['color_image'] = color_image
		shm_e2fgvi['depth_image'] = depth_image
		shm_e2fgvi['frame_no'] = frame_no
		shm_e2fgvi['mask_occluder'] = mask_occluder

		break

from multiprocessing import Manager
h, w = 640, 360
zeros_color = np.zeros((h, w, 3), dtype=np.uint8)
zeros_gray = np.zeros((h, w), dtype=np.uint8)
zeros_bool = np.zeros((h, w), dtype=bool)

class Cache:
	def __init__(self, max_size=0):
		self.__cache = [None] * max_size if max_size > 0 else [None]
		self.max_size = max_size
		self.__pointer = -1

	def put(self, value):
		if self.max_size > 0:
			self.__pointer += 1
			if self.__pointer == self.max_size:
				self.__pointer = 0

			self.__cache[self.__pointer] = value
		else:
			self.__cache.append(value)

	def get_all(self):
		return [value for value in self.__cache if value is not None]

	def get_latest(self):
		if self.max_size > 0:
			return self.__cache[self.__pointer]
		else:
			return self.__cache[-1]

	def get_oldest(self):
		if self.max_size > 0:
			p = self.__pointer - 1
			if p < 0:
				p = self.max_size-1
			return self.__cache[p]
		else:
			return self.__cache[0]

	def any(self):
		return len(self.get_all()) > 0


if __name__ == '__main__':
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
		shm_sa['mask_hand_cache'] = Cache(max_size=2)
		shm_sa['mask_hand_retrieved'] = None
		shm_sa['mask_hand_retrieved_cache'] = Cache(max_size=5)
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

		# E2FGVI
		shm_e2fgvi = manager.dict()
		shm_e2fgvi['color_image'] = zeros_color
		shm_e2fgvi['depth_image'] = zeros_gray
		shm_e2fgvi['frame_no'] = 0
		shm_e2fgvi['mask_occluder'] = zeros_bool
		shm_e2fgvi['color_image_pil_cache'] = Cache(10)
		shm_e2fgvi['mask_occluder_pil_cache'] = Cache(10)

		e2fgvi_task(shm_sa, shm_e2fgvi, shm_flags)