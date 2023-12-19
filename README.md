# 使用方法
## セットアップ
```
conda create -n ofht python=3.11
conda activate ofht
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

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