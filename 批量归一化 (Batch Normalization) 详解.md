# 批量归一化 (Batch Normalization) 详解

> **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**  
> 作者: Sergey Ioffe & Christian Szegedy (2015)  
> 影响: 深度学习最重要的创新之一

---

## 目录
1. [核心概念](#一核心概念)
2. [数学原理](#二数学原理)
3. [算法流程](#三算法流程)
4. [训练 vs 推理](#四训练-vs-推理)
5. [代码实现](#五代码实现)
6. [优点与缺点](#六优点与缺点)
7. [使用技巧](#七使用技巧)
8. [与其他归一化对比](#八与其他归一化对比)
9. [总结](#九总结)

---

## 一、核心概念

### 1.1 什么是 Batch Normalization？

**批量归一化 (BN)** 是一种在深度神经网络中对**每层的激活值进行归一化**的技术，使其保持稳定的分布（均值接近0，方差接近1）。

### 1.2 为什么需要 BN？

#### 问题：内部协变量偏移 (Internal Covariate Shift)

```
深度网络训练过程中，每层输入的分布不断变化：

┌────────────────────────────────────────────┐
│ 没有 BN 的网络                              │
├────────────────────────────────────────────┤
│ Input      → Layer 1 → Layer 2 → Layer 3   │
│ [0~255]      [-10,50]   [-100,500]  爆炸!  │
│                                             │
│ 问题:                                       │
│ • 分布不断偏移                              │
│ • 梯度不稳定                                │
│ • 训练困难                                  │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ 使用 BN 的网络                              │
├────────────────────────────────────────────┤
│ Input → BN → Layer 1 → BN → Layer 2 → BN   │
│ [0~255]  ≈N(0,1)  ≈N(0,1)     ≈N(0,1)      │
│                                             │
│ 效果:                                       │
│ • 分布保持稳定                              │
│ • 梯度稳定                                  │
│ • 训练顺利                                  │
└────────────────────────────────────────────┘
```

---

## 二、数学原理

### 2.1 完整的 4 步计算

对于一个 mini-batch 的数据 **B = {x₁, x₂, ..., xₘ}**：

#### **步骤 1: 计算均值 (Mean)**

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

#### **步骤 2: 计算方差 (Variance)**

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$

#### **步骤 3: 归一化 (Normalize)**

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

其中 **ε** (epsilon) 是一个小常数（如 1e-5），防止除零。

#### **步骤 4: 缩放和平移 (Scale and Shift)**

$$
y_i = \gamma \hat{x}_i + \beta
$$

其中：
- **γ** (gamma): 可学习的缩放参数
- **β** (beta): 可学习的偏移参数
- 每个通道有独立的 γ 和 β

### 2.2 为什么需要 γ 和 β？

```
目的: 恢复网络的表达能力

如果只做归一化:
  强制所有特征 mean=0, std=1
  可能损害网络性能

引入可学习参数后:
  y = γ · x̂ + β
  
特殊情况:
  如果 γ = σ, β = μ
  则 y = x (完全恢复原始分布)
  
网络可以学习:
  • 是否需要归一化
  • 归一化到什么程度
  • 最优的分布参数
```

### 2.3 对于卷积层的计算

对于 4D 张量 **[N, C, H, W]**：

```
N: Batch size (批量大小)
C: Channels (通道数)
H: Height (高度)
W: Width (宽度)

归一化维度: (N, H, W)
独立统计量: C 组（每个通道一组）

对于通道 c:
  μc = (1/NHW) Σ(n,h,w) x[n,c,h,w]
  σc² = (1/NHW) Σ(n,h,w) (x[n,c,h,w] - μc)²
  
每个通道有独立的:
  • γc (scale)
  • βc (shift)
```

---

## 三、算法流程

### 3.1 伪代码

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    x: 输入 [batch_size, channels, height, width]
    gamma: 缩放参数 [channels]
    beta: 偏移参数 [channels]
    """
    
    # 1. 计算 mini-batch 的均值和方差
    mu = mean(x, axis=(0, 2, 3))      # 对 (N, H, W) 求均值
    var = variance(x, axis=(0, 2, 3))  # 对 (N, H, W) 求方差
    
    # 2. 归一化
    x_hat = (x - mu) / sqrt(var + eps)
    
    # 3. 缩放和平移
    y = gamma * x_hat + beta
    
    return y
```

### 3.2 反向传播

```python
梯度计算（链式法则）:

∂L/∂γ = Σ(∂L/∂y · x̂)
∂L/∂β = Σ(∂L/∂y)
∂L/∂x̂ = ∂L/∂y · γ
∂L/∂σ² = Σ[∂L/∂x̂ · (x - μ) · (-1/2) · (σ² + ε)^(-3/2)]
∂L/∂μ = Σ[∂L/∂x̂ · (-1/√(σ² + ε))] + ∂L/∂σ² · (2/m)Σ(x - μ) · (-1)
∂L/∂x = ∂L/∂x̂ · (1/√(σ² + ε)) + ∂L/∂σ² · (2/m)(x - μ) + ∂L/∂μ · (1/m)
```

---

## 四、训练 vs 推理

### 4.1 训练模式 (Training Mode)

```python
训练时使用当前 mini-batch 的统计量:

1. 计算当前 batch 的 μ 和 σ²
2. 用于归一化当前 batch
3. 更新移动平均统计量（用于推理）

移动平均更新:
  running_mean = momentum × running_mean + (1-momentum) × μ_batch
  running_var = momentum × running_var + (1-momentum) × σ²_batch
  
  默认 momentum = 0.1 (PyTorch)
```

### 4.2 推理模式 (Inference Mode)

```python
推理时使用训练时积累的移动平均统计量:

1. 使用 running_mean 和 running_var
2. 不计算当前 batch 的统计量
3. 保证单样本推理的稳定性

推理公式:
  x̂ = (x - running_mean) / √(running_var + ε)
  y = γ · x̂ + β
```

### 4.3 模式切换

```python
import torch.nn as nn

model = MyModel()

# 训练模式
model.train()
# BN 使用 batch 统计量，更新 running stats

# 推理模式
model.eval()
# BN 使用 running stats，不更新

⚠️ 重要: 必须显式切换模式！
```

---

## 五、代码实现

### 5.1 PyTorch 标准使用

```python
import torch
import torch.nn as nn

# 对于 2D 卷积 (图像)
bn_2d = nn.BatchNorm2d(
    num_features=64,          # 通道数
    eps=1e-5,                 # 防止除零
    momentum=0.1,             # 移动平均动量
    affine=True,              # 使用可学习的 γ 和 β
    track_running_stats=True  # 跟踪移动平均
)

# 对于 1D (序列/全连接)
bn_1d = nn.BatchNorm1d(num_features=128)

# 使用示例
input = torch.randn(32, 64, 28, 28)  # [N, C, H, W]
output = bn_2d(input)

# 查看参数
print(f"γ (weight): {bn_2d.weight.shape}")        # [64]
print(f"β (bias): {bn_2d.bias.shape}")            # [64]
print(f"running_mean: {bn_2d.running_mean.shape}") # [64]
print(f"running_var: {bn_2d.running_var.shape}")   # [64]
```

### 5.2 标准 CNN 块

```python
class ConvBNReLU(nn.Module):
    """标准的 Conv-BN-ReLU 块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size//2,
            bias=False  # ⚠️ BN 会抵消 bias，设为 False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)    # 归一化
        x = self.relu(x)  # 激活
        return x

# 使用
block = ConvBNReLU(64, 128)
input = torch.randn(16, 64, 56, 56)
output = block(input)
print(output.shape)  # [16, 128, 56, 56]
```

### 5.3 ResNet 风格残差块

```python
class ResidualBlock(nn.Module):
    """ResNet 残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out
```

### 5.4 手动实现 BN

```python
class MyBatchNorm2d(nn.Module):
    """手动实现的 BatchNorm2d"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
        # 移动平均统计量（不参与梯度计算）
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x):
        # x: [N, C, H, W]
        
        if self.training:
            # 训练模式：使用 batch 统计量
            mu = x.mean(dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            
            # 归一化
            x_hat = (x - mu) / torch.sqrt(var + self.eps)
            
            # 更新移动平均
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * var
                self.num_batches_tracked += 1
        else:
            # 推理模式：使用移动平均统计量
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # 缩放和平移
        y = self.gamma * x_hat + self.beta
        
        return y
```

---

## 六、优点与缺点

### 6.1 优点 ✅

#### 1. **加速训练**

```
效果: 训练速度提升 2-10 倍

原因:
• 归一化使梯度更稳定
• 可以使用更大的学习率
• 减少训练轮数

对比:
  没有 BN: 100 epochs, lr=0.01
  使用 BN:  30 epochs, lr=0.1
  时间节省 70%
```

#### 2. **允许更大的学习率**

```
没有 BN:
  learning_rate = 0.01  (小心翼翼)
  
使用 BN:
  learning_rate = 0.1-1.0  (大胆使用)
  
原因: BN 使梯度更稳定，不容易梯度爆炸
```

#### 3. **减少对初始化的敏感性**

```
没有 BN:
  需要精心设计权重初始化
  • Xavier initialization
  • He initialization
  
使用 BN:
  对初始化不敏感
  甚至随机初始化也能训练
```

#### 4. **正则化效果**

```
BN 引入轻微的噪声:
  • 每个 mini-batch 的统计量略有不同
  • 类似于轻微的 Dropout 效果
  
实践中:
  使用 BN 后可以减小或去除 Dropout
```

#### 5. **允许使用饱和激活函数**

```
没有 BN:
  • Sigmoid/Tanh 容易梯度消失
  • 必须使用 ReLU
  
使用 BN:
  • 可以安全使用 Sigmoid/Tanh
  • BN 保证输入在合理范围
```

### 6.2 缺点 ❌

#### 1. **依赖 Batch Size**

```
┌─────────────┬──────────┬────────────┐
│ Batch Size  │ BN 效果  │ 建议        │
├─────────────┼──────────┼────────────┤
│ ≥ 32        │ ✅ 很好  │ 推荐使用 BN │
│ 16-31       │ ⚠️ 尚可  │ 可以使用 BN │
│ 8-15        │ ❌ 较差  │ 考虑 GN     │
│ < 8         │ ❌ 很差  │ 用 GN/LN    │
└─────────────┴──────────┴────────────┘

原因: batch 太小时，统计量不准确
```

#### 2. **训练和推理不一致**

```
训练: 使用 batch 统计量
推理: 使用移动平均统计量

可能导致:
  • 训练时性能好，推理时性能下降
  • 分布不匹配 (train-test mismatch)
  
注意: 必须显式切换 model.train() / model.eval()
```

#### 3. **不适合 RNN/序列模型**

```
RNN 问题:
  • 序列长度可变
  • 时间步之间需要独立统计量
  • BN 在 RNN 中效果不好
  
解决方案:
  使用 Layer Normalization (LN)
```

#### 4. **增加内存和计算开销**

```
额外开销:
  • 需要存储 running_mean 和 running_var
  • 推理时需要额外的归一化计算
  • 反向传播更复杂
  
缓解:
  • BN Folding (融合到前一层)
  • 量化部署时可以移除 BN
```

#### 5. **分布式训练的挑战**

```
问题:
  每个 GPU 的 batch 太小
  
解决方案:
  • Synchronized BN (多 GPU 同步统计量)
  • Group Normalization
```

---

## 七、使用技巧

### 7.1 何时使用 BN？

```python
✅ 推荐使用:
  • 图像分类 (ResNet, VGG, DenseNet)
  • 目标检测 (Faster R-CNN)
  • 图像分割 (U-Net, DeepLab)
  • Batch size ≥ 16
  • CNN 架构

❌ 不推荐使用:
  • Batch size 很小 (< 8)
  • RNN / LSTM / GRU
  • Transformer 模型
  • GAN 的判别器
  • 在线学习 (单样本)
  • 强化学习
```

### 7.2 BN 的位置

```python
# 标准顺序: Conv → BN → Activation ✅ 推荐
x = conv(x)
x = bn(x)
x = relu(x)

# 原因:
# 1. 在激活前归一化，保证激活函数的输入分布稳定
# 2. 大多数论文和实现都采用这个顺序
# 3. ResNet 官方实现就是这样

# 注意: Conv 层不需要 bias
conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
# 因为 BN 会减去均值，bias 会被抵消
```

### 7.3 微调预训练模型

```python
# 场景: 使用 ImageNet 预训练模型微调小数据集

# 方法 1: 冻结 BN 统计量 ✅ 推荐（小数据集）
model = torchvision.models.resnet50(pretrained=True)
model.train()

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()  # 保持推理模式
        # 可选: 冻结参数
        module.weight.requires_grad = False
        module.bias.requires_grad = False

# 方法 2: 使用小学习率更新 BN
conv_params = [p for name, p in model.named_parameters() 
               if 'bn' not in name]
bn_params = [p for name, p in model.named_parameters() 
             if 'bn' in name]

optimizer = torch.optim.SGD([
    {'params': conv_params, 'lr': 0.01},
    {'params': bn_params, 'lr': 0.001}  # BN 用更小的学习率
])
```

### 7.4 BN Folding (推理优化)

```python
# BN Folding: 将 BN 融合到前一层卷积
# 推理时可以加速，减少内存

def fuse_conv_bn(conv, bn):
    """将 Conv + BN 融合为一个 Conv"""
    
    # 获取参数
    w_conv = conv.weight.data
    
    # BN 参数
    gamma = bn.weight.data
    beta = bn.bias.data
    running_mean = bn.running_mean.data
    running_var = bn.running_var.data
    eps = bn.eps
    
    # 融合公式
    # y = γ · (conv(x) - μ) / σ + β
    #   = (γ/σ) · conv(x) + (β - γ·μ/σ)
    
    std = torch.sqrt(running_var + eps)
    w_fused = w_conv * (gamma / std).view(-1, 1, 1, 1)
    b_fused = beta - gamma * running_mean / std
    
    # 创建融合后的卷积层
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=True
    )
    fused_conv.weight.data = w_fused
    fused_conv.bias.data = b_fused
    
    return fused_conv

# 使用
# fused_conv = fuse_conv_bn(conv, bn)
```

### 7.5 BN 的超参数

```python
bn = nn.BatchNorm2d(
    num_features=64,
    
    eps=1e-5,  # 通常不需要改
    # 太小: 可能数值不稳定
    # 太大: 影响归一化效果
    
    momentum=0.1,  # PyTorch 默认
    # 0.1: 新 batch 的权重为 10%
    # 越小: 统计量更新越慢，更平滑
    # 越大: 统计量更新越快，可能不稳定
    
    affine=True,  # 通常保持 True
    # True: 使用可学习的 γ 和 β
    # False: 强制 γ=1, β=0（很少用）
    
    track_running_stats=True  # 通常保持 True
    # True: 跟踪 running_mean 和 running_var
    # False: 训练和推理都用 batch 统计（很少用）
)
```

---

## 八、与其他归一化对比

### 8.1 四种归一化方法

```
┌──────────┬─────────────┬─────────────┬──────────────┬────────────┐
│ 方法      │ 归一化维度   │ Batch依赖   │ 适用场景     │ 代表网络   │
├──────────┼─────────────┼─────────────┼──────────────┼────────────┤
│ Batch    │ (N,H,W)     │ ✅ 是       │ CV,大batch   │ ResNet     │
│ Norm     │ 跨 batch    │             │              │ DenseNet   │
├──────────┼─────────────┼─────────────┼──────────────┼────────────┤
│ Layer    │ (C,H,W)     │ ❌ 否       │ NLP,RNN      │ Transformer│
│ Norm     │ 跨 channel  │             │              │ BERT,GPT   │
├──────────┼─────────────┼─────────────┼──────────────┼────────────┤
│ Instance │ (H,W)       │ ❌ 否       │ 风格迁移     │ CycleGAN   │
│ Norm     │ 单 channel  │             │ 图像生成     │ StyleGAN   │
├──────────┼─────────────┼─────────────┼──────────────┼────────────┤
│ Group    │ (C/G,H,W)   │ ❌ 否       │ 目标检测     │ Mask R-CNN │
│ Norm     │ 组内        │             │ 小 batch     │ YOLO       │
└──────────┴─────────────┴─────────────┴──────────────┴────────────┘

维度说明:
N: Batch size
C: Channels
H: Height
W: Width
G: Groups
```

### 8.2 可视化对比

```
输入张量: [N=2, C=4, H=3, W=3]

Batch Norm:
━━━━━━━━━━━━━━━━━━━━━━━━
对每个 channel，跨 (N, H, W) 归一化
每个 channel 一组统计量 (共 4 组)

Layer Norm:
━━━━━━━━━━━━━━━━━━━━━━━━
对每个样本，跨 (C, H, W) 归一化
每个样本一组统计量 (共 2 组)

Instance Norm:
━━━━━━━━━━━━━━━━━━━━━━━━
对每个样本的每个 channel，跨 (H, W) 归一化
每个样本每个 channel 一组统计量 (共 2×4=8 组)

Group Norm (G=2):
━━━━━━━━━━━━━━━━━━━━━━━━
对每个样本的每组 channels，跨组内 (C/G, H, W) 归一化
每个样本每组一组统计量 (共 2×2=4 组)
```

### 8.3 选择指南

```python
if task == "图像分类" and batch_size >= 16:
    use BatchNorm
elif task == "NLP" or model == "Transformer":
    use LayerNorm
elif task == "风格迁移" or task == "图像生成":
    use InstanceNorm
elif task == "目标检测" or batch_size < 16:
    use GroupNorm
elif task == "视频理解" or memory_limited:
    use GroupNorm
```

---

## 九、总结

### 9.1 核心要点

```
┌─────────────────────────────────────────────┐
│ Batch Normalization 核心总结                 │
├─────────────────────────────────────────────┤
│                                              │
│ 📐 数学公式:                                 │
│   1. μ = (1/m) Σ xi                         │
│   2. σ² = (1/m) Σ (xi - μ)²                 │
│   3. x̂ = (x - μ) / √(σ² + ε)                │
│   4. y = γ·x̂ + β                            │
│                                              │
│ 🎯 主要作用:                                 │
│   • 加速训练 (2-10倍)                        │
│   • 允许更大学习率                           │
│   • 减少初始化敏感性                         │
│   • 轻微的正则化效果                         │
│   • 稳定深度网络训练                         │
│                                              │
│ ⚙️ 实现要点:                                 │
│   • 训练用 batch 统计量                      │
│   • 推理用移动平均统计量                     │
│   • 每个通道独立的 γ 和 β                   │
│   • Conv 层可以去掉 bias                    │
│   • 通常放在激活函数前                       │
│                                              │
│ 📊 归一化维度:                               │
│   • 2D: 跨 (N, H, W)                        │
│   • 1D: 跨 N                                │
│   • 每个 channel 有独立统计量               │
│                                              │
│ ⚠️ 注意事项:                                 │
│   • 需要较大 batch size (≥16)               │
│   • 训练/推理模式要切换                      │
│   • 不适合 RNN/Transformer                   │
│   • 小 batch 用 Group Norm                   │
│   • 微调时考虑冻结 BN                        │
│                                              │
│ 🚀 性能影响:                                 │
│   • 训练加速: 2-10x                          │
│   • 学习率提升: 10-100x                      │
│   • 收敛速度: 显著提升                       │
│   • 最终精度: 通常提升 1-3%                  │
│                                              │
└─────────────────────────────────────────────┘
```

### 9.2 使用建议

```
✅ 什么时候用 BN:
  • 图像分类、检测、分割
  • CNN 架构
  • Batch size ≥ 16
  • 需要快速训练
  • 深层网络 (> 10 层)

❌ 什么时候不用 BN:
  • Batch size < 8
  • RNN / LSTM
  • Transformer (用 LN)
  • GAN 判别器
  • 在线学习
  • 强化学习

🔄 替代方案:
  • 小 batch → Group Norm
  • RNN → Layer Norm
  • 风格迁移 → Instance Norm
  • Transformer → Layer Norm
```

### 9.3 历史影响

```
Batch Normalization (2015) 的影响:

革命性贡献:
✅ 使训练深度网络变得容易
✅ 催生了更深的网络架构
✅ 成为几乎所有 CNN 的标配
✅ 启发了其他归一化方法

代表性网络:
• ResNet (2015) - 152 层，得益于 BN
• Inception v2/v3 (2015) - 使用 BN
• DenseNet (2017) - 使用 BN
• EfficientNet (2019) - 使用 BN

后续发展:
• Layer Norm (2016) - NLP 领域
• Group Norm (2018) - 小 batch
• Weight Normalization (2016)
• Spectral Normalization (2018) - GAN
```

### 9.4 关键论文

```
必读论文:

1. 原始论文:
   Batch Normalization: Accelerating Deep Network Training 
   by Reducing Internal Covariate Shift
   Ioffe & Szegedy, ICML 2015
   
2. 深入分析:
   How Does Batch Normalization Help Optimization?
   Santurkar et al., NeurIPS 2018
   (质疑 ICS 假设，提出新的解释)
   
3. 其他归一化:
   Group Normalization
   Wu & He, ECCV 2018
   
   Layer Normalization
   Ba et al., arXiv 2016
```

---

## 十、快速参考

### 10.1 PyTorch 速查

```python
import torch.nn as nn

# 2D (图像)
bn2d = nn.BatchNorm2d(channels)

# 1D (序列/全连接)
bn1d = nn.BatchNorm1d(features)

# 3D (视频)
bn3d = nn.BatchNorm3d(channels)

# 模式切换
model.train()  # 训练模式
model.eval()   # 推理模式

# 冻结 BN
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
```

### 10.2 常见问题

```
Q1: BN 放在激活前还是激活后？
A1: 通常放在激活前 (Conv → BN → ReLU)

Q2: 为什么 Conv 可以不要 bias？
A2: BN 会减去均值，bias 会被抵消，用 BN 的 β 代替

Q3: Batch size 多大合适？
A3: ≥16 较好，≥32 最佳，< 8 考虑 Group Norm

Q4: 推理时为什么不用 batch 统计量？
A4: 推理可能是单样本，统计量不准确

Q5: 微调预训练模型要冻结 BN 吗？
A5: 小数据集建议冻结，大数据集可以不冻结

Q6: BN 能用在 RNN 吗？
A6: 不推荐，用 Layer Normalization

Q7: 训练和推理结果不一致怎么办？
A7: 确保调用了 model.eval()
```

---

**文档创建时间**: 2025-11-16  
**版本**: 1.0  
**参考**: Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015

---

**相关文档**:
- [GPU结构详解.md](./GPU结构详解.md)
- [归一化层对比.md](./归一化层对比.md)
- [深度学习优化技巧.md](./深度学习优化技巧.md)

