# 情感计算

## Clip来将图像和表情对齐

### 使用方法：

下载rafdb数据集，放在Tip adapter/data文件夹

main.py不要用，用修改后的main_mod.py

大部分的配置修改都可以在configs/rafdb.yaml里完成

### 一套比较好的参数（可以在configs/rafdb.yaml里改）

1 探索不同的Prompt
以下是512 shot下的表现：
[{$label}]:
Best Acc: 81.03, Acc: 81.78
[The expression of the person in the picture is {}.]:
Best Acc: 82.11, Acc: 82.76
['The expression is {}.']:
Best Acc: 82.07, Acc: 82.72

2 探索少样本的个数（Best Acc是模型遵循Initial Beta/Alpha默认设置下跑出的最好测试集精度，Acc是超参数搜索后自动优化得到的最优测试集精度）
16：Best Acc: 72.39, Acc: 72.59 LR:0.001
32：Best Acc: 74.05, Acc: 74.38 LR: 0.001
64：Best Acc: 78.00, Acc: 78.32 LR: 0.0006
128：Best Acc: 79.95, Acc: 80.25 LR: 0.0008
256：Best Acc: 80.31, Acc: 81.26 LR: 0.0001
512: Best Acc: 81.03, Acc: 81.78 LR: 0.00004
全部：Acc: 79.07 LR: 0.006 （感觉是因为只能微调Adapter，模型容量不够以很好拟合整个数据集，也曾将学习率调小，epoch数增加到1000，但此时测试集最佳精度只有75.55）

以上所有结果都是设定为30个epoch时完成的，Shot的选择遵循原论文中的生成少样本数据集的方法

3 探索微调adapter以外的层
微调visual模块：78%左右（使用了Balanced CrossEntropy策略）

原始Tip-Adapter作者的README在original_README.md
