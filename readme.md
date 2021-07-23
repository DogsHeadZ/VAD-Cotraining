# 编码计划

之前的代码已经上传到这个仓库内[https://github.com/DogsHeadZ/VAD-Cotraining](https://github.com/DogsHeadZ/VAD-Cotraining)，首先做好代码清洗，把**所有重复的和没用的代码都删了包括config文件**，只保留程序运行的主逻辑，即一个dataloader，一个train和一个evaluate和其余必要工具。

**先做这个**，代码规范必要写点注释，程序训练和测试都能跑通后（可先用avenue或ped2测试），push到你的lxy分支内，写个较为详细的运行步骤和命令（包括光流图的生成和bbox的生成等等），我之后会clone你的分支。

## 1、dataloader

输出：

还是基于预测的框架，输入连续T帧上的objects及其对应的光流图。即返回两个tensor，分别为rgb tensor大小为[time_step+num_pred, objects个数, 图片的通道数, _resize_height, _resize_width]和optical tensor大小为[time_step+num_pred, objects个数, 光流的通道数, _resize_height, _resize_width]。这次最好处理好batchsize不为1的情况，加快训练速率。

光流部分或许可以参考这篇[https://github.com/wanboyang/anomly_feature.pytorch](https://github.com/wanboyang/anomly_feature.pytorch)，也可以先用我们之前的。

## 2、自编码器

包含rgb自编码器和flow自编码器，输入分别为dataloader返回的两个tensor。

自编码器的代码像之前一样可以用PreAE.py也可以用networks.py，看哪个效果好用哪个。

这里就能完成训练和测试了，看实验结果定下自编码器的网络。

## 3、对比损失

这里需要将自编码器的特征降维为1D，可以采用映射函数（或池化操作）对其进行降维，但**解码器是解码降维前的特征还是降维后的特征？**（这里需要进行实验来确定）

## 4、分类器

这里我想了下，直接对编码器的特征进行分类效果应该会不好。因为最新的那两篇半监督的VAD都是通过I3D提取16帧的特征，给这16帧当作整体打一个分数（这16帧每帧的分数都一样）。所以感觉也得用这种做法，每16帧提取特征进行分类，毕竟分类网络直接输出分数而自编码器还得计算重构误差。

## 5、协同训练

暂且不写。

## 6、Spatial Transformer

这部分我看看，暂且不写。

## 7、I3D分支主逻辑

包括10crops数据增强，这部分我来做，我近期把那两篇论文的代码理解下并跑通，然后放到我们的模型里。

https://github.com/wanboyang/anomly_feature.pytorch



