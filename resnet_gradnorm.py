import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# CIFAR-10 数据集加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 定义模型（这里使用ResNet18作为示例）
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        self.shared_layers = models.resnet50()
        weight = torch.load("./weight/resnet50-19c8e357.pth")
        self.shared_layers.load_state_dict(weight)
        num_features = self.shared_layers.fc.in_features
        self.shared_layers.fc = nn.Identity()  # 移除全连接层

        # 为两个任务定义独立的全连接层
        self.task1_fc = nn.Linear(num_features, 10)
        self.task2_fc = nn.Linear(num_features, 10)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        task1_output = self.task1_fc(shared_output)
        task2_output = self.task2_fc(shared_output)
        return task1_output, task2_output


model = MultiTaskResNet().to('cuda')
criterion = nn.CrossEntropyLoss()


# 定义 GradNorm
class GradNorm:
    def __init__(self, model, alpha=0.5):
        self.alpha = alpha
        self.model = model
        self.task_weights = torch.ones(2, requires_grad=True, device='cuda')
        self.initial_losses = None

    def compute_gradients(self, losses):
        # 清零梯度
        self.model.zero_grad()
        # 获取各任务损失的梯度
        grads = []
        for i, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            grad_norm = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += (param.grad.norm() ** 2)
            grads.append(torch.sqrt(grad_norm))
        return grads

    def update_weights(self, losses):
        grads = self.compute_gradients(losses)
        if self.initial_losses is None:
            self.initial_losses = [loss.item() for loss in losses]

        # 计算各任务的加权损失
        loss_ratios = [loss / init_loss for loss, init_loss in zip(losses, self.initial_losses)]
        mean_loss_ratio = sum(loss_ratios) / len(loss_ratios)

        # 计算梯度比率
        grad_target = [loss_ratio ** self.alpha * mean_loss_ratio for loss_ratio in loss_ratios]
        grad_norm = torch.stack(grads)
        grad_norm_target = torch.tensor(grad_target, device='cuda', requires_grad=True)

        # 计算加权损失
        weight_losses = torch.abs(grad_norm - grad_norm_target).sum()
        #weighted_loss.requires_grad_(True)
        weight_losses.backward()


# 定义优化器
optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': GradNorm(model).task_weights, 'lr': 0.001}
], lr=0.001)

# 训练过程
num_epochs = 10
grad_norm = GradNorm(model)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_loss_task1 = 0  # 用于记录任务1的损失
    total_loss_task2 = 0  # 用于记录任务2的损失
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to('cuda'), labels.to('cuda')

        # 前向传播
        task1_output, task2_output = model(images)

        # 分别计算每个任务的损失
        loss1 = criterion(task1_output, labels)
        loss2 = criterion(task2_output, labels)
        losses = [loss1, loss2]

        # 使用 GradNorm 更新权重
        grad_norm.update_weights(losses)

        # 将各任务损失加权求和
        weighted_loss = (grad_norm.task_weights[0] * loss1 + grad_norm.task_weights[1] * loss2)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()
        total_loss_task1 += loss1.item()  # 累加任务1的损失
        total_loss_task2 += loss2.item()  # 累加任务2的损失

        # 输出训练进度（每隔一定步数）
        if (i + 1) % 10 == 0:  # 每100个batch输出一次
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                  f'Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, '
                  f'Total Loss: {total_loss / (i + 1):.4f}, Loss1 (Avg): {total_loss_task1 / (i + 1):.4f}, '
                  f'Loss2 (Avg): {total_loss_task2 / (i + 1):.4f}')

    # 每个 epoch 结束后输出平均损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss / len(train_loader):.4f}, '
          f'Avg Loss1: {total_loss_task1 / len(train_loader):.4f}, Avg Loss2: {total_loss_task2 / len(train_loader):.4f}')

print("训练完成")

