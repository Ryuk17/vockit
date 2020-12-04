#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: KDD
# @time: 2018-11-10
"""
![aukit](aukit.png "aukit")

## aukit
audio toolkit: 语音和频谱处理的工具箱。

### 安装

```
pip install -U aukit
```

- 注意
    * 可能需另外安装的依赖包：tensorflow, pyaudio, sounddevice。
    * tensorflow<=1.13.1
    * pyaudio暂不支持python37以上版本直接pip安装，需要下载whl文件安装，下载路径：https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
    * sounddevice依赖pyaudio。
    * aukit的默认音频采样率为16k。
"""

__version__ = '1.4.4'

from .audio_io import load_wav, save_wav, anything2bytesio, anything2wav, anything2bytes, Dict2Obj, _sr
from .audio_spectrogram import linear_spectrogram, mel_spectrogram
from .audio_spectrogram import default_hparams as hparams_spectrogram
from .audio_spectrogram import linear2mel_spectrogram, mel2linear_spectrogram
from .audio_griffinlim import inv_linear_spectrogram, inv_linear_spectrogram_tf, inv_mel_spectrogram
from .audio_griffinlim import default_hparams as hparams_griffinlim
from .audio_io import __doc__ as io_doc
from .audio_spectrogram import __doc__ as spectrogram_doc
from .audio_griffinlim import __doc__ as griffinlim_doc

version_doc = """
### 版本
v{}
""".format(__version__)

readme_docs = [__doc__, version_doc, io_doc, griffinlim_doc, spectrogram_doc]

if __name__ == "__main__":
    print(__file__)
