import torch

# w, h = 1280, 720
# w, h = 848, 480
w, h = 640, 360

input_fps = 30
input_seconds_per_frame = 1 / input_fps

input_from_file = True
input_filepath = r'record/ex7_p2_1.mp4'

save_result = False
record_in_video_cv2 = False
record_in_video_ffmpeg = False
record_in_images = False

device = torch.device("cuda")