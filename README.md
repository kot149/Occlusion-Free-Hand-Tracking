# 使用方法
## セットアップ
```
git clone https://github.com/kot149/Occlusion-Free-Hand-Tracking.git
cd Occlusion-Free-Hand-Tracking

cd src
git clone https://github.com/MCG-NKU/E2FGVI.git
cd ..

conda create -n ofht2 python=3.8
conda activate ofht2

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install -r requirements.txt
```
<!-- ```
# without E2FGVI
conda create -n ofht python=3.8
conda activate ofht
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
``` -->

## 実行
```
python src/ofht.py
```

## 確認済みの動作環境
- Windows 11
- Python 3.8 - 3.11
- Nvidia GeForce RTX 4070
- GeForce Game Ready Driver 546.29
- CUDA 12.2
- cuDNN 8.9
- Intel Core i7-13700F
- Memory 32GB