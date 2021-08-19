# 代码更新记录

## 1、dataloader

### 7.29更新

光流部分用的还是之前的，[https://github.com/wanboyang/anomly_feature.pytorch](https://github.com/wanboyang/anomly_feature.pytorch)里面找不到光流提取的代码，好像这个仓库的代码不是很全

目标检测部分重构了一下代码，在 `getROI.py`中，使用方法：

```python
# 初始化
object_detector = ObjectDetector(args.weight, args.gpu) 
# 获取某个帧的目标位置
roi = object_detector.getRoI(img_path)
# 如果输入多个帧的话，默认提取最后一帧的目标位置
roi = object_detector.getRoi([img1_path, img2_path, img3_path])
```

这部分删除了之前结合motion的部分，实在是太不稳定了。

(注释了`yolov5/utils/datasets.py`中181行左右的print，不然总是输出图片的路径，太烦了)

**TODO**

目标检测部分还是有些问题，比如在ShanghaiTech这个数据集里面，总是把井盖给框出来，感觉没有必要。如果有人打伞的话，会把人和伞分别框出来。（可能会考虑筛选一下类别，只保留人的类别，但是还没研究过UCF-crime中的异常事件是不是和物体有关）

16帧的时间长度，人已经完全走开了

#### dataloader

vad_dataloader_object.py

返回的是[ objects个数,time_step+num_pred, 图片的通道数, _resize_height, _resize_width] （这里我把objects个数放在第一个维度了，如果要把t放第一维也可以）

因为每个帧的object数量会不一样，为了应对batch_size不为1的情况，进行了填补，补成最大的object数目

dataloader测试：

```
python vad_dataloader_object.py --datadir ./dataset/ShanghaiTech/Training/frames
```

#### 一些文件名的修改

models文件夹改成了model，utils.py改成了my_utils.py

主要是yolov5文件夹下也有这两个文件，文件多了互相import的时候会出bug

#### 训练测试

写了一个训练的过程测试一下，大致为输入一帧的object，然后输入PreAE进行重构。(测试的时候我batchsize设的是4)

```
python train_AE.py --config configs/lyx_trainAE.yaml --gpu 0
```

训练的过程是跑通了，如果要改成预测或者改成多帧的话也比较方便改。

val的代码没有写

## 7.30

添加了光流

train_AE.py调试完成

#### TODO

bug：如果batch中所有的frame都没有object会出错

制作.h5文件

## 8.16
fix bug：如果所有的frame中都没有object会出错（这里就是简单地判断了一下如果没有的话就跳过这次训练）

制作.h5文件的代码`make_h5.py`，运行方式：`python make_h5.py`

h5文件：key为videoname-framenum，data为裁剪出的16帧中所有的目标的rgb和光流

train_AE.py是一个测试文件，没有什么实际的训练意义，就是构建了一个rgb的preAE和flow的preAE，然后裁剪16帧的所有object和他们的flow来训练自编码器。用的时候batch_size可以不为1，但是不能设得太大，因为如果遇到某帧的object特别多，显存就容易爆，或者可以 --gpu 0,1,2,3。

从[MIST_VAD/make_h5.py at master · fjchange/MIST_VAD (github.com)](https://github.com/fjchange/MIST_VAD/blob/master/utils/make_h5.py)的make_h5.py文件中来看的话，他好像没有用滑窗，就是16帧16帧切分，sample的数量应该是比我们少很多

##### todo

还没有处理好h5文件，看了一下MIST代码ShanghaiTech的h5文件有30多G，我们应该要大很多，后续是否要按照它这样处理？如果这样处理的话一旦有什么错误，重新生成这个文件需要很久。

**在半监督学习的时候是给每个object打伪标签还是给每个frame打伪标签？（这个不是很清楚所以还没有写train_AE.py的后半部分）**

## 8.17

分别构建基于rgb和flow的自编码器，用于重构每一帧中所有object的rgb和光流，测试时将帧中所有的目标重构误差（rgb重构误差和光流重构误差之和）最大的作为这一帧的误差，如果没有object则mse误差为0，得到测试集的AUC。

```
python train_AE.py --config configs/lyx_trainAE.yaml --gpu 0,1
```

但是AUC不是很高，目前最高的AUC只有63.7%

## 8.19

更新了计算flow并保存h5

使用opencv来计算光流（优点是任意尺寸的输入都可以，缺点是慢，这里我保存成和图片一样的大小）

```
python make_h5_opencvflow.py
```

使用flownet2.0来计算光流（优点是快，缺点是不支持任意尺寸的输入，这里我保存成256*256）

```
python make_h5_flow.py --gpu 0
```

