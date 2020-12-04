# vockit
Vocoder Toolkit. 声码器工具箱。


### 声码器列表

1. griffinlim
2. waveglow
3. melgan
4. wavernn
5. wavenet
6. parallelwavegan
7. squeezewave
8. lpcnet
9. gansynth
10. autorange
11. spsi
12. world

### 声码器来源

1. [griffinlim](https://github.com/keithito/tacotron/tree/master/util)
2. [waveglow](https://github.com/NVIDIA/waveglow)
3. [melgan](https://github.com/descriptinc/melgan-neurips)
4. [wavernn](https://github.com/fatchord/WaveRNN)
5. [wavenet](https://github.com/r9y9/wavenet_vocoder)
6. [parallelwavegan](https://github.com/kan-bayashi/ParallelWaveGAN)
7. [squeezewave](https://github.com/tianrengao/SqueezeWave)
8. [lpcnet](https://github.com/mozilla/LPCNet)
9. [gansynth](https://github.com/ss12f32v/GANsynth-pytorch)
10. [auorange](https://github.com/Yablon/auorange)
11. [spsi](https://github.com/lonce/SPSI_Python)
12. [world](https://github.com/KuangDD/aukit)

### 简要介绍

1. griffinlim

- 纯信号处理，不用训练。
- 使用线性频谱或梅尔频谱。
- 使用高维度的线性频谱生成的语音音质相对较高，使用梅尔频谱则声音较干巴巴。
- 生成声音效果稳定，极少出现不可控问题。


2. waveglow

- 基于非自回归神经网络模型，由一个网络构成，不需要自回归的过程，用一个损失函数进行训练，简单有效。
- 一般使用梅尔频谱。
- 预训练模型对中文语音的生成，逼真度较高，出现少数少许和原声不相似的情况，不相似的听感较好，平滑过渡。
- 总体生成声音效果稳定。

3. melgan

- 基于GAN实现的，整体结构不难理解就是由生成器和判别器组成。
- 一般使用梅尔频谱。
- 预训练模型对中文语音的生成，大多发音人逼真度高，存在部分发音人不太相似，不相似的部分听感粗糙，音质明显下降。
- 稳定性存在不可控因素，对特定发音人的合成会使得可控性提高。

4. wavernn
5. wavenet
6. parallelwavegan
7. squeezewave
8. lpcnet
9. gansynth
10. autorange
11. spsi
12. world


### TODO
- 先列个声码器的列表，重点只搞几个声码器。
- 弄个统一规范的频谱生成方式，频谱转换方式，适配多个声码器。
- 弄个统一规范的使用声码器的方法，一键直达。
- 构建可直接pip安装的package。
