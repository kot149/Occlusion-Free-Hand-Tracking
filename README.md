# システムの概要
- 入力: RGBD(RGB+Depth)映像
  - Intel RealSense Depth Camera D455で撮影
- 出力: 手に被っているモノ(手のオクルーダー)のマスク

(最初のフレームのみ)
1. MediaPipeとDepth画像を用いて、手のバウンディングボックスを求める
   - MediaPipeのランドマーク座標の分散とDepth値から、ボックスのサイズを決める
   - MediaPipeのランドマークから手の中心を決め、その周りに正方形を描く
2. 手のバウンディングボックス内でSegment Anything(の高速化版のFastSAM)を実行
3. MediaPipeを用いて、Segment Anythingの結果から手のマスクを取り出す

(2フレーム目以降)
1. 前のフレームの手のマスクとDepth画像を用いて手のバウンディングボックスを求める
   - Depth値からボックスのサイズを決める
   - 前のフレームの手のマスクとバウンディングボックスの余白に応じてバウンディングボックスの位置を調整
     - 単純に重心を中心に取ると、手首の方が伸びていってしまうのでNG
2. 手のバウンディングボックス内でSegment Anything(の高速化版のFastSAM)を実行
3. 前のフレームの手のマスクとの類似度(IoU)を用いて、Segment Anythingの結果から手のマスクを取り出す (手のマスクの追従)
4. 手のマスクの前のフレームとの差分を取る
5. Segment Anythingの結果のうち、差分と被っていて、かつ手よりDepth値が小さい(=手よりカメラに近い)ものを取り出し、オクルーダーのマスク(出力)とする

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