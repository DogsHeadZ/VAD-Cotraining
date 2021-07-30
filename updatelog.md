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


