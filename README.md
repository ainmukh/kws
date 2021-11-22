# KWS project: CRNN compression and speed up

## Installation guide
```shell
git clone https://github.com/ainmukh/kws
cd kws
pip install -q -r ./requirements.txt
```

### Warning
If using colab, you must do the following as colab does not support torch==1.10.0 yet.
```shell
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Get baseline and split
```shell
mkdir saved
wget https://www.dropbox.com/s/etbywu0xvvccwsl/indices.pth
wget https://www.dropbox.com/s/tvr7bq7kj7ct00k/model_base.pth
mv indices.pth model_base.pth
```
## Get all models
```shell
mkdir saved
wget https://www.dropbox.com/sh/dg7ofsb1xollq49/AACpI2Fk9pw9qtwmP5j8WxNUa
unzip AACpI2Fk9pw9qtwmP5j8WxNUa -d saved
```
