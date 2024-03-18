import matplotlib.pyplot as plt
import numpy as np

def visualize_masks(masks):
	h, w = masks[0].shape[:2]
	result = np.zeros((h, w, 3), dtype=np.uint8)
	for m in masks:
		color = (np.random.random(3)*255).astype(np.uint8)
		result[m, :] = color

	return result

from collections import deque
import time
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

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2

model_type = "vit_t"
sam_checkpoint = "model_checkpoint\mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=5, points_per_batch=120)


img = cv2.imread(r"D:\Google Drive\KIT\lab\hand-tracking\Occlusion-Free-Hand-Tracking\output\ex7_p1_1_\rgb\00180.png")

# predictor = SamPredictor(mobile_sam)
# predictor.set_image(img)
# masks, _, _ = predictor.predict(<input_prompts>)

fps_counter = Fps_Counter()
for i in range(50):
	masks = mask_generator.generate(img)
	print(fps_counter.count())

masks_seg = [m['segmentation'] for m in masks]
result = visualize_masks(masks_seg)

plt.imshow(result)
