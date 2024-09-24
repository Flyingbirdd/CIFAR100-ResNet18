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




## Requirements

This is my experiment eviroument
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


## Training Details
I didn't use any training tricks to improve accuray, if you want to learn more about training tricks,
please refer to my another [repo](https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks), contains
various common training tricks and their pytorch implementations.


I follow the hyperparameter settings in paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.1 divide by 5 at 60th, 120th, 160th epochs, train for 200
epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9. You could also use the hyperparameters from paper [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548v1) and [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2), which is initial lr = 0.1, lr divied by 10 at 150th and 225th epochs, and training for 300 epochs with batchsize 128, this is more commonly used. You could decrese the batchsize to 64 or whatever suits you, if you dont have enough gpu memory.

You can choose whether to use TensorBoard to visualize your training procedure

## Results
The result I can get from a certain model, since I use the same hyperparameters to train all the networks, some networks might not get the best result from these hyperparameters, you could try yourself by finetuning the hyperparameters to get
better result.

|dataset|network|params|top1 err|top5 err|epoch(lr = 0.1)|epoch(lr = 0.02)|epoch(lr = 0.004)|epoch(lr = 0.0008)|total epoch|
|:-----:|:-----:|:----:|:------:|:------:|:-------------:|:--------------:|:---------------:|:----------------:|:---------:|
|cifar100|mobilenet|3.3M|34.02|10.56|60|60|40|40|200|
|cifar100|mobilenetv2|2.36M|31.92|09.02|60|60|40|40|200|
|cifar100|squeezenet|0.78M|30.59|8.36|60|60|40|40|200|
|cifar100|shufflenet|1.0M|29.94|8.35|60|60|40|40|200|
|cifar100|shufflenetv2|1.3M|30.49|8.49|60|60|40|40|200|
|cifar100|vgg11_bn|28.5M|31.36|11.85|60|60|40|40|200|
|cifar100|vgg13_bn|28.7M|28.00|9.71|60|60|40|40|200|
|cifar100|vgg16_bn|34.0M|27.07|8.84|60|60|40|40|200|
|cifar100|vgg19_bn|39.0M|27.77|8.84|60|60|40|40|200|
|cifar100|resnet18|11.2M|24.39|6.95|60|60|40|40|200|
|cifar100|resnet34|21.3M|23.24|6.63|60|60|40|40|200|
|cifar100|resnet50|23.7M|22.61|6.04|60|60|40|40|200|
|cifar100|resnet101|42.7M|22.22|5.61|60|60|40|40|200|
|cifar100|resnet152|58.3M|22.31|5.81|60|60|40|40|200|
|cifar100|preactresnet18|11.3M|27.08|8.53|60|60|40|40|200|
|cifar100|preactresnet34|21.5M|24.79|7.68|60|60|40|40|200|
|cifar100|preactresnet50|23.9M|25.73|8.15|60|60|40|40|200|
|cifar100|preactresnet101|42.9M|24.84|7.83|60|60|40|40|200|
|cifar100|preactresnet152|58.6M|22.71|6.62|60|60|40|40|200|
|cifar100|resnext50|14.8M|22.23|6.00|60|60|40|40|200|
|cifar100|resnext101|25.3M|22.22|5.99|60|60|40|40|200|
|cifar100|resnext152|33.3M|22.40|5.58|60|60|40|40|200|
|cifar100|attention59|55.7M|33.75|12.90|60|60|40|40|200|
|cifar100|attention92|102.5M|36.52|11.47|60|60|40|40|200|
|cifar100|densenet121|7.0M|22.99|6.45|60|60|40|40|200|
|cifar100|densenet161|26M|21.56|6.04|60|60|60|40|200|
|cifar100|densenet201|18M|21.46|5.9|60|60|40|40|200|
|cifar100|googlenet|6.2M|21.97|5.94|60|60|40|40|200|
|cifar100|inceptionv3|22.3M|22.81|6.39|60|60|40|40|200|
|cifar100|inceptionv4|41.3M|24.14|6.90|60|60|40|40|200|
|cifar100|inceptionresnetv2|65.4M|27.51|9.11|60|60|40|40|200|
|cifar100|xception|21.0M|25.07|7.32|60|60|40|40|200|
|cifar100|seresnet18|11.4M|23.56|6.68|60|60|40|40|200|
|cifar100|seresnet34|21.6M|22.07|6.12|60|60|40|40|200|
|cifar100|seresnet50|26.5M|21.42|5.58|60|60|40|40|200|
|cifar100|seresnet101|47.7M|20.98|5.41|60|60|40|40|200|
|cifar100|seresnet152|66.2M|20.66|5.19|60|60|40|40|200|
|cifar100|nasnet|5.2M|22.71|5.91|60|60|40|40|200|
|cifar100|wideresnet-40-10|55.9M|21.25|5.77|60|60|40|40|200|
|cifar100|stochasticdepth18|11.22M|31.40|8.84|60|60|40|40|200|
|cifar100|stochasticdepth34|21.36M|27.72|7.32|60|60|40|40|200|
|cifar100|stochasticdepth50|23.71M|23.35|5.76|60|60|40|40|200|
|cifar100|stochasticdepth101|42.69M|21.28|5.39|60|60|40|40|200|



