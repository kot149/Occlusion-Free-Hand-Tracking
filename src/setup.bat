pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -U opencv-python
pip install ffmpeg-python

@REM RealSense
pip install pyrealsense2
@REM apt-get install lsb-core
@REM mkdir -p /etc/apt/keyrings
@REM curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
@REM echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
@REM tee /etc/apt/sources.list.d/librealsense.list
@REM apt-get update
@REM apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg

@REM MediaPipe
pip install mediapipe

@REM FastSAM
pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git

@REM VSCode Extensions
@REM code --install-extension formulahendry.code-runner
@REM ms-python.python
