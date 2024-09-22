import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from sam import SAM


# 训练函数定义
def train(epoch):
    start = time.time()  # 记录开始时间
    net.train()  # 设置模型为训练模式
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:  # 如果使用GPU
            labels = labels.cuda()  # 将标签加载到GPU
            images = images.cuda()  # 将图像加载到GPU

        # 定义SAM优化器所需的closure函数
        def closure():
            optimizer.zero_grad()  # 梯度清零，防止累积梯度
            outputs = net(images)  # 前向传播，获取模型输出
            loss = loss_function(outputs, labels)  # 计算损失值
            loss.backward()  # 反向传播，计算梯度
            return loss  # 返回损失

        # SAM优化器的第一步 (SAM step 1)
        loss = closure()  # 计算损失并进行反向传播
        optimizer.first_step(zero_grad=True)  # 进行SAM的第一步优化

        # SAM优化器的第二步 (SAM step 2)，无需再次计算梯度
        closure()  # 再次执行前向传播，但不计算梯度
        optimizer.second_step(zero_grad=True)  # 进行SAM的第二步优化

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1  # 迭代步数

        # 使用TensorBoard记录最后一层参数的梯度信息
        last_layer = list(net.children())[-1]  # 获取网络的最后一层
        for name, para in last_layer.named_parameters():
            if para.grad is not None:  # 确保参数有梯度
                if 'weight' in name:  # 如果是权重参数
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:  # 如果是偏置参数
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        # 输出训练进度
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],  # 当前学习率
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),  # 当前已经训练的样本数
            total_samples=len(cifar100_training_loader.dataset)  # 数据集总样本数
        ))

        # 在TensorBoard中记录每一次迭代的损失
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:  # 如果在warm-up阶段
            warmup_scheduler.step()  # 更新学习率

    # 在TensorBoard中记录每一层参数的直方图
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()  # 记录结束时间
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))  # 输出训练时间


# 评估函数定义（无需计算梯度）
@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()  # 记录开始时间
    net.eval()  # 设置模型为评估模式

    test_loss = 0.0  # 初始化损失
    correct = 0.0  # 初始化正确预测数

    for (images, labels) in cifar100_test_loader:

        if args.gpu:  # 如果使用GPU
            images = images.cuda()  # 将图像加载到GPU
            labels = labels.cuda()  # 将标签加载到GPU

        outputs = net(images)  # 前向传播
        loss = loss_function(outputs, labels)  # 计算损失

        test_loss += loss.item()  # 累加测试集的损失
        _, preds = outputs.max(1)  # 获取预测值
        correct += preds.eq(labels).sum()  # 计算正确预测的个数

    finish = time.time()  # 记录结束时间
    if args.gpu:  # 输出GPU信息
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),  # 计算平均损失
        correct.float() / len(cifar100_test_loader.dataset),  # 计算准确率
        finish - start  # 评估所用时间
    ))
    print()

    # 如果启用TensorBoard，则记录评估的损失和准确率
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)  # 返回准确率


# 主程序入口
if __name__ == '__main__':

    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='网络类型')  # 例如'resnet18'
    parser.add_argument('-gpu', action='store_true', default=False, help='是否使用GPU')
    parser.add_argument('-b', type=int, default=128, help='dataloader的批量大小')
    parser.add_argument('-warm', type=int, default=1, help='warm-up训练阶段的epoch数量')
    parser.add_argument('-lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('-resume', action='store_true', default=False, help='是否恢复训练')
    args = parser.parse_args()

    net = get_network(args)  # 获取网络模型

    # 数据预处理与加载：
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,  # CIFAR-100数据集的均值
        settings.CIFAR100_TRAIN_STD,  # CIFAR-100数据集的标准差
        num_workers=4,  # 读取数据的线程数量
        batch_size=args.b,  # 批量大小
        shuffle=True  # 是否打乱数据
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 初始化SAM优化器，使用SGD作为基础优化器
    base_optimizer = torch.optim.SGD
    optimizer = SAM(base_optimizer=base_optimizer, params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 学习率调度器，使用MultiStepLR进行学习率衰减
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)  # 定义warm-up学习率调度器

    if args.resume:  # 如果选择恢复训练
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('没有找到最近的文件夹')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:  # 否则新建一个检查点文件夹
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # 使用TensorBoard记录
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # TensorBoard不能覆盖旧的值，必须创建新日志文件
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)  # 输入一个示例张量以创建模型图
    if args.gpu:
        input_tensor = input_tensor.cuda()  # 将输入张量加载到GPU
    writer.add_graph(net, input_tensor)  # 将网络图添加到TensorBoard

    # 创建检查点文件夹以保存模型
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0  # 记录最优准确率
    if args.resume:  # 如果恢复训练，加载之前的模型参数
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('找到最佳精度的权重文件:{}'.format(weights_path))
            print('加载最佳训练文件以测试精度...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)  # 评估当前模型精度
            print('最佳精度为 {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('没有找到最近的权重文件')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('加载权重文件 {} 以恢复训练.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))  # 记录恢复的epoch数

    # 开始训练循环
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:  # 如果超过warm-up阶段
            train_scheduler.step(epoch)  # 更新学习率

        if args.resume:  # 如果恢复训练，跳过已完成的epoch
            if epoch <= resume_epoch:
                continue

        train(epoch)  # 训练模型
        acc = eval_training(epoch)  # 评估模型

        # 当学习率衰减到0.01后，保存性能最好的模型
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('保存权重文件到 {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)  # 保存最佳模型权重
            best_acc = acc  # 更新最佳精度
            continue

        # 定期保存模型
        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('保存权重文件到 {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)  # 保存常规权重

    writer.close()  # 关闭TensorBoard记录器
writer.close()