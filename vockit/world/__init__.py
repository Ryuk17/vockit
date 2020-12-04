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
from .audio_world import world_spectrogram, inv_world_spectrogram, change_voice
from .audio_world import world_spectrogram_default, inv_world_spectrogram_default

from .audio_io import __doc__ as io_doc
from .audio_world import __doc__ as world_doc

version_doc = """
### 版本
v{}
""".format(__version__)

readme_docs = [__doc__, version_doc, io_doc, world_doc]

if __name__ == "__main__":
    print(__file__)
