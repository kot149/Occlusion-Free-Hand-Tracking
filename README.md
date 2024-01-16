# システムの概要
- 入力: RGBD(RGB+Depth)映像
  - Intel RealSense Depth Camera D455で撮影
- 出力: 手に被っているモノ(手のオクルーダー)のマスク

(最初のフレームのみ)
1. MediaPipeを用いて、手のバウンディングボックスを求める
   - MediaPipeのランドマーク座標の分散とDepth値から、ボックスのサイズを決める
   - MediaPipeのランドマークから手の中心を決め、その周りに正方形を描く
2. 手のバウンディングボックス内でSegment Anything(の高速化版のFastSAM)を実行
3. MediaPipeのランドマークの座標を用いて、Segment Anythingの結果から手のマスクを取り出す
4. 手のマスクでTrack-Anythingを開始する

(2フレーム目以降)
1. Track-Anythingで手のマスクを取り出す
2. 手のマスクから手のバウンディングボックスを決める
3. 手のバウンディングボックス内でSegment Anything(の高速化版のFastSAM)を実行
4. Segment Anythingの結果のうち、手のマスクと最も似ているものをIoUにより判定し、Segment Anythingの結果から取り除く
5. Segment Anythingの結果のうち手より手前にありそうなマスクを、マスク内のdepth値の最大値により判定して求める
6. モルフォロジー変換の膨張処理を用いて手のマスクのエッジを取る
7. 手より手前にありそうなマスクのうち、手のマスクのエッジとの共通部分があり、かつその共通部分のdepth値が手のマスクより小さい(手より近い)ものを、オクルーダーのマスクとする

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