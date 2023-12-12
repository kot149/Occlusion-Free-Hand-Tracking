import pyrealsense2 as rs
import numpy as np
import cv2
import time

class Fps_Counter:
	def __init__(self, bufsize=30):
		self.fps = -1
		self._count = -1
		self._bufsize = bufsize
		self.start()

	def start(self):
		self._buf = np.zeros(self._bufsize)
		self._timestamp = time.time()


	def count(self):
		self._count += 1
		if self._count == self._bufsize:
			self._count = 0

		t = time.time()
		self._buf[self._count] = t - self._timestamp
		self._timestamp = t

		ave = np.mean(self._buf[self._buf != 0])
		self.fps = 1 / ave

		return self.fps

# ストリーミングの設定
# w, h = 640, 480
w, h = 848, 480
fps = 60

config = rs.config()
config.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
config.enable_stream(rs.stream.infrared, 2, w, h, rs.format.y8, fps)
config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

# ストリーミング開始
pipeline = rs.pipeline()
pipeline.start(config)

fps_counter = Fps_Counter()

try:
	while True:
		# フレーム待ち
		frames = pipeline.wait_for_frames()

		#IR１
		ir_frame1 = frames.get_infrared_frame(1)
		ir_image1 = np.asanyarray(ir_frame1.get_data())

		#IR2
		ir_frame2 = frames.get_infrared_frame(2)
		ir_image2 = np.asanyarray(ir_frame2.get_data())

		# RGB
		color_frame = frames.get_color_frame()
		color_image = np.asanyarray(color_frame.get_data())

		# 深度
		depth_frame = frames.get_depth_frame()
		depth_image = np.asanyarray(depth_frame.get_data())

		# 2次元データをカラーマップに変換
		ir_colormap1   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image1), cv2.COLORMAP_JET)
		ir_colormap2   = cv2.applyColorMap(cv2.convertScaleAbs(ir_image2), cv2.COLORMAP_JET)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)

		# イメージの結合
		images = np.vstack(( np.hstack((ir_colormap1, ir_colormap2)), np.hstack((color_image, depth_colormap)) ))

		# 表示
		cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('RealSense', images)

		# FPS
		fps = fps_counter.count()
		print(fps, 'fps')

		# q キー入力で終了
		if cv2.waitKey(1) == 27:
			cv2.destroyAllWindows()
			break

finally:
	# ストリーミング停止
	pipeline.stop()