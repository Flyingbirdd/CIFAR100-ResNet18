# Pytorch-cifar100

practice on cifar100 using pytorch
项目简介
论文<Escaping Saddle Points for Effective Generalizationon Class-Imbalanced Data >发现，锐度感知最小化Sharpness-Aware Minimization (SAM) 通过重新加权可以有效地增强沿负曲率的梯度分量，从而有效地摆脱鞍点，从而提高泛化性能。在针对长尾学习和类不平衡学习设计的各种重加权和裕度增强方法中，SAM可以显著提高性能。
该项目基于深度学习训练神经网络，使用 CIFAR-100 数据集进行图像分类任务，并通过使用SAM优化器提升模型的泛化能力。项目使用了 PyTorch 框架进行模型构建和训练，包含了数据预处理、模型训练、评估以及模型保存的完整流程。
1. 数据集与数据加载
项目使用 CIFAR-100 数据集，包含100类彩色图像。项目通过 torchvision.transforms 对图像进行标准化处理，并将训练集与测试集分别加载到 DataLoader 中，便于批量处理。
2. 模型架构
项目支持自定义网络架构，用户通过 -net 参数指定不同的网络类型。模型通过 get_network 函数创建，支持GPU加速训练（通过 -gpu 参数开启）。
3. 损失函数与优化器
使用 交叉熵损失函数（CrossEntropyLoss） 来衡量模型输出与真实标签之间的误差。优化器采用了 SAM (Sharpness-Aware Minimization)，其特点是能够通过双步优化的方式最小化参数邻域内的最大损失，提高模型在不同数据分布上的鲁棒性。SAM优化器在每个训练批次内分两步执行：
•	第一步更新权重以找到较陡峭的梯度方向。
•	第二步使用该方向调整权重，提高模型的泛化性能。
4. 训练过程
在每个训练周期（epoch）中，模型进行前向传播计算输出，并使用损失函数计算误差。采用SAM优化器的双步更新策略，分别在第一步和第二步更新模型参数。此外，通过 TensorBoard 记录梯度变化、损失值等信息，便于可视化和分析。项目使用了 MultiStepLR 学习率调度器，按预设的里程碑（milestones）降低学习率，从而促进模型在训练后期更好地收敛。训练前期还包含了学习率预热（Warm-up）阶段，逐步提高学习率以避免梯度爆炸。
5. 模型评估
在每个 epoch 结束后，模型会在测试集上进行评估，计算平均损失和分类准确率。测试过程不进行梯度更新，通过 torch.no_grad() 减少计算资源的占用。评估结果同样会记录到 TensorBoard。
6. 模型保存与恢复
模型在每个 epoch 结束后，按预设条件保存到指定的文件夹中。项目支持从断点恢复训练，用户可以通过 -resume 参数加载最近一次保存的模型权重和训练状态，从而继续训练。
7. 项目效果
项目核心优化器 SAM 的工作原理是通过双步优化来寻找损失平滑的方向，从而提高泛化能力。通过使用 SAM 优化器，该项目旨在提升模型在 CIFAR-100 数据集上的准确率，特别是在数据分布变化或未知噪声干扰下的泛化性能。项目还提供了训练与评估的可视化工具，便于监控训练过程中的重要指标（如梯度、损失、准确率等）。
8. 训练结果：
   ![6c29126cf5d9f13f6f2c50eac3bff75](https://github.com/user-attachments/assets/eb8cb501-ab25-4b53-92bb-ab1200881ade)
   ![2fe7fb74eac0972f62e0b850dfe8238](https://github.com/user-attachments/assets/67c051d9-19cb-474d-8652-a73b8cff9977)
| Step | Value  |
|------|--------|
| 1    | 0.1301 |
| 11   | 0.5124 |
| 21   | 0.5623 |
| 31   | 0.5916 |
| 41   | 0.5759 |
| 51   | 0.6220 |
| 61   | 0.7364 |
| 71   | 0.7240 |
| 81   | 0.7147 |
| 91   | 0.7140 |
| 101  | 0.7118 |
| 111  | 0.7113 |
| 121  | 0.7570 |
| 131  | 0.7611 |
| 141  | 0.7620 |
| 151  | 0.7622 |
| 161  | 0.7662 |
| 171  | 0.7689 |
| 181  | 0.7658 |
| 191  | 0.7674 |
| 200  | 0.7685 |

模型训练的三个阶段：

初始阶段：模型的性能逐渐提升，说明开始有效学习。
中间阶段：尽管有波动，模型整体表现继续改善，并逐步适应更复杂的特征。
后期阶段：性能稳定在高水平，波动减少，表明模型达到了良好的效果。
训练过程中模型不断优化，最终表现稳定且达到较高水平。


## Requirement
 experiment eviroument：
- python3.6
- pytorch1.6.0+cu101
- tensorboard 2.2.2(optional)


## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. dataset
I will use cifar100 dataset from torchvision since it's more convenient, but I also
kept the sample code for writing your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. run tensorbard(optional)
Install tensorboard
```bash
$ pip install tensorboard
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 4. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train resnet18
$ python train.py -net resnet18 -gpu
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.

 the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).


### 5. test the model
Test the model using test.py
```bash
$ python test.py -net resnet18 -weights path_to_resnet18_weights_file
```

