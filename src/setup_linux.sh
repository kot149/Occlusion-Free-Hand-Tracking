# libusb
apt-get update && apt-get install -y libgl1-mesa-dev && apt-get -y install libusb-1.0-0-dev


# for RealSense
apt-get install lsb-core
mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
tee /etc/apt/sources.list.d/librealsense.list
apt-get update
apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg
