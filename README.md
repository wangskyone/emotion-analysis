# emotion-analysis

### Dataset

#### rafdb
http://www.whdeng.cn/raf/model1.html
#### GENKI-4K
https://github.com/watersink/GENKI

### TODO


### 二、基于 GENKI-4K 笑脸识别数据集的分类任务
  源代码见`wangqingyu/face.py`

#### 2.1 GENKI-4K
  发布于2009年，GENKI数据集是由加利福尼亚大学的机器概念实验室收集。该数据集包含GENKI-R2009a，GENKI-4K，GENKI-SZSL三个部分。GENKI-R2009a包含11159个图像，GENKI-4K包含4000个图像，分为“笑”和“不笑”两种，每个图片拥有不同的尺度大小，姿势，光照变化，头部姿态，可专门用于做笑脸识别。这些图像包括广泛的背景，光照条件，地理位置，个人身份和种族等。本实验选用GENKI-4K数据集，按照9:1分为训练集和测试集。  
  数据集链接 `https://github.com/watersink/GENKI`

#### 2.2 模型设计
  采用卷积神经、池化、全连接层，包括以下几个步骤：  
  `conv1 = torch.nn.Conv2d(3, 10, kernel_size = 3)`、`pooling1 = torch.nn.MaxPool2d(2)`、`ReLU`、  
  `conv2 = torch.nn.Conv2d(10, 20, kernel_size = 3)`、`pooling2 = torch.nn.MaxPool2d(2)`、`ReLU`、  
  `Reshape`、`fc1 = torch.nn.Linear(3920, 512)`、`ReLU`、`fc2 = torch.nn.Linear(512, 2)`。  
  `fc2`的输出做为最终模型的输出。
  
#### 2.3 实验设计
  采用CrossEntropyLoss()损失函数、SGD优化、初始学习率为0.01,、动量因子0.5、Epochs为100。
  
#### 2.4 运行
  `python == 3.8.8`  
  `torch == 1.13.0`  
  运行实验  `nohup python face.py &`

#### 2.5 实验结果
  50个Epoch之后模型收敛，达到85%的准确率，最高准确率为86%。
