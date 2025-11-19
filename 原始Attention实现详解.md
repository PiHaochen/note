# 原始 Attention 实现详解

## 目录
1. [Attention 机制概述](#一attention-机制概述)
2. [数学公式](#二数学公式)
3. [实现步骤详解](#三实现步骤详解)
4. [内存层次与数据流动](#四内存层次与数据流动)
5. [性能瓶颈分析](#五性能瓶颈分析)
6. [代码实现](#六代码实现)
7. [实际案例分析](#七实际案例分析-llama3-8b)
8. [优化方向](#八优化方向)

---

## 一、Attention 机制概述

Attention（注意力机制）是 Transformer 架构的核心组件，用于捕捉序列中不同位置之间的依赖关系。

### 核心思想
- 对于序列中的每个位置，计算它与所有其他位置的相关性（注意力权重）
- 基于这些权重对值向量进行加权求和
- 使模型能够"关注"序列中最相关的部分

### 输入输出
- **输入**：Query (Q), Key (K), Value (V) 三个矩阵
- **输出**：加权后的表示 O

---

## 二、数学公式

### 标准 Attention 公式

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 参数说明

| 符号 | 含义 | 形状 |
|------|------|------|
| $Q$ | Query（查询）矩阵 | $[N, d]$ |
| $K$ | Key（键）矩阵 | $[N, d]$ |
| $V$ | Value（值）矩阵 | $[N, d]$ |
| $N$ | 序列长度 | 标量 |
| $d$ | 特征维度（每个头） | 标量 |
| $S$ | 注意力分数矩阵 $QK^T$ | $[N, N]$ |
| $P$ | 注意力权重矩阵（Softmax后） | $[N, N]$ |
| $O$ | 输出矩阵 | $[N, d]$ |

### 计算步骤分解

1. **计算注意力分数**：$S = QK^T$
2. **缩放**：$S' = \frac{S}{\sqrt{d_k}}$（防止梯度消失）
3. **归一化**：$P = \text{Softmax}(S')$
4. **加权求和**：$O = PV$

---

## 三、实现步骤详解

### 原始 Attention 的 10 个步骤

假设 $Q, K, V \in \mathbb{R}^{N \times d}$ 存储在 **HBM（High Bandwidth Memory，显存）**。

```
┌─────────────────────────────────────────────────────────────┐
│                    原始 Attention 流程                        │
└─────────────────────────────────────────────────────────────┘

HBM (显存)              SRAM (片上内存)              操作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

步骤 1: 从 HBM 加载 Q, K 到 SRAM
   Q, K  ────────────>  Q, K                      [HBM → SRAM]

步骤 2: 计算 S = QK^T
                        Q, K  ──→  S              [矩阵乘法]

步骤 3: 将 S 写回到 HBM
   S  <────────────────  S                        [SRAM → HBM]

步骤 4: 将 S 加载到 SRAM
   S  ────────────────>  S                        [HBM → SRAM]

步骤 5: 计算 P = Softmax(S)
                        S  ──→  P                 [Softmax]

步骤 6: 将 P 写回到 HBM
   P  <────────────────  P                        [SRAM → HBM]

步骤 7: 从 HBM 加载 P 和 V 到 SRAM
   P, V  ──────────────>  P, V                    [HBM → SRAM]

步骤 8: 计算 O = PV
                        P, V  ──→  O              [矩阵乘法]

步骤 9: 将 O 写回到 HBM
   O  <────────────────  O                        [SRAM → HBM]

步骤 10: 返回 O
```

### 详细说明

#### 步骤 1：加载 Q, K
- **操作**：从 HBM 读取 Q 和 K 矩阵到 SRAM
- **数据量**：$2Nd$ 个元素
- **时间**：慢（HBM 访问延迟高）

#### 步骤 2：计算注意力分数
- **操作**：$S = QK^T$
- **计算量**：$O(N^2d)$ 次浮点运算
- **结果**：$S \in \mathbb{R}^{N \times N}$
- **关键问题**：S 的大小是 $N^2$，当 N 很大时，SRAM 无法容纳

#### 步骤 3：写回 S
- **操作**：将 S 从 SRAM 写回 HBM
- **数据量**：$N^2$ 个元素
- **原因**：SRAM 容量有限（通常几十到几百 KB）

#### 步骤 4：重新加载 S
- **操作**：从 HBM 读取 S 到 SRAM
- **数据量**：$N^2$ 个元素
- **问题**：刚写回去又要读回来，造成冗余的 HBM 访问

#### 步骤 5：计算 Softmax
- **操作**：对 S 的每一行做 Softmax
- **公式**：$P_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^N \exp(S_{ik})}$
- **计算**：
  ```
  对于每一行 i:
    1. 找到最大值 m_i = max(S_i)
    2. 计算 exp(S_ij - m_i)（数值稳定性）
    3. 求和 sum_i = Σ exp(S_ij - m_i)
    4. 归一化 P_ij = exp(S_ij - m_i) / sum_i
  ```

#### 步骤 6：写回 P
- **操作**：将 P 从 SRAM 写回 HBM
- **数据量**：$N^2$ 个元素

#### 步骤 7：加载 P 和 V
- **操作**：从 HBM 读取 P 和 V 到 SRAM
- **数据量**：$N^2 + Nd$ 个元素

#### 步骤 8：计算输出
- **操作**：$O = PV$
- **计算量**：$O(N^2d)$ 次浮点运算
- **结果**：$O \in \mathbb{R}^{N \times d}$

#### 步骤 9：写回 O
- **操作**：将最终结果 O 写回 HBM
- **数据量**：$Nd$ 个元素

#### 步骤 10：完成
- 返回输出矩阵 O

---

## 四、内存层次与数据流动

### GPU 内存层次

```
┌─────────────────────────────────────────────────────────┐
│                    GPU 内存层次                          │
├─────────────────────────────────────────────────────────┤
│  寄存器 (Registers)                                      │
│  - 速度: ~1 cycle                                        │
│  - 容量: 几 KB / 线程                                    │
│  - 作用域: 线程私有                                      │
├─────────────────────────────────────────────────────────┤
│  共享内存 (Shared Memory / SRAM)                         │
│  - 速度: ~5 cycles                                       │
│  - 容量: 48-96 KB / Block                                │
│  - 作用域: Block 内共享                                  │
├─────────────────────────────────────────────────────────┤
│  L1 Cache                                                │
│  - 速度: ~30 cycles                                      │
│  - 容量: 几十 KB / SM                                    │
├─────────────────────────────────────────────────────────┤
│  L2 Cache                                                │
│  - 速度: ~200 cycles                                     │
│  - 容量: 几 MB                                           │
├─────────────────────────────────────────────────────────┤
│  全局内存 / HBM (High Bandwidth Memory)                  │
│  - 速度: ~400-800 cycles                                 │
│  - 容量: 几 GB - 几十 GB                                 │
│  - 作用域: 全局                                          │
└─────────────────────────────────────────────────────────┘
```

### 速度对比

| 内存类型 | 延迟 | 相对速度 |
|---------|------|---------|
| 寄存器 | 1x | 800x |
| Shared Memory | 5x | 160x |
| L1 Cache | 30x | 27x |
| L2 Cache | 200x | 4x |
| HBM | 800x | 1x |

**关键点**：HBM 比 Shared Memory 慢约 **160 倍**！

### 原始 Attention 的 HBM 访问次数

| 步骤 | 操作 | 读取 | 写入 | 数据量 |
|------|------|------|------|--------|
| 1 | 加载 Q, K | ✓ | | $2Nd$ |
| 3 | 写回 S | | ✓ | $N^2$ |
| 4 | 加载 S | ✓ | | $N^2$ |
| 6 | 写回 P | | ✓ | $N^2$ |
| 7 | 加载 P, V | ✓ | | $N^2 + Nd$ |
| 9 | 写回 O | | ✓ | $Nd$ |
| **总计** | | | | $2N^2 + 4Nd$ |

**IO 复杂度**：$O(N^2)$

---

## 五、性能瓶颈分析

### 问题 1：中间矩阵占用 $O(N^2)$ 空间

- **S 矩阵**：$N \times N$
- **P 矩阵**：$N \times N$
- **总空间**：$2N^2$ 个浮点数

#### 实际数据（Llama3-8B）
- 序列长度 $N = 8192$
- $N^2 = 67,108,864$ 个元素
- 单精度（float32）：$67M \times 4 \text{ bytes} = 268 \text{MB}$
- 32 个注意力头：$268 \text{MB} \times 32 = 8.6 \text{GB}$

### 问题 2：频繁的 HBM 访问

- S 和 P 矩阵需要**写回再读取**
- 每次 HBM 访问延迟 ~800 个时钟周期
- 大量时间浪费在等待数据传输

### 问题 3：计算与 IO 不平衡

```
计算时间 vs IO 时间
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

理想情况（计算密集）:
  [████████████████████████████████] 计算
  [█] IO

实际情况（IO 密集）:
  [████] 计算
  [████████████████████████████████] IO 等待
```

**结论**：GPU 的计算能力没有充分利用，大部分时间在等待数据。

### 问题 4：序列长度的平方增长

| 序列长度 N | S + P 大小 | 内存占用（float32） |
|-----------|-----------|-------------------|
| 512 | 262K | 1 MB |
| 1024 | 1M | 4 MB |
| 2048 | 4M | 16 MB |
| 4096 | 16M | 64 MB |
| 8192 | 67M | 268 MB |
| 16384 | 268M | 1 GB |
| 32768 | 1B | 4 GB |

**问题**：序列长度翻倍，内存占用增加 4 倍！

---

## 六、代码实现

### 6.1 PyTorch 实现（高层）

```python
import torch
import torch.nn.functional as F
import math

def naive_attention(Q, K, V, mask=None):
    """
    原始 Attention 实现
    
    参数:
        Q: Query 矩阵 [batch, seq_len, dim]
        K: Key 矩阵 [batch, seq_len, dim]
        V: Value 矩阵 [batch, seq_len, dim]
        mask: 可选的掩码 [batch, seq_len, seq_len]
    
    返回:
        O: 输出矩阵 [batch, seq_len, dim]
        P: 注意力权重 [batch, seq_len, seq_len]
    """
    batch_size, seq_len, dim = Q.shape
    
    # 步骤 1-2: 计算注意力分数 S = QK^T
    # Q: [batch, seq_len, dim]
    # K^T: [batch, dim, seq_len]
    # S: [batch, seq_len, seq_len]
    S = torch.matmul(Q, K.transpose(-2, -1))
    
    # 缩放（防止梯度消失）
    S = S / math.sqrt(dim)
    
    # 应用掩码（可选，用于因果注意力）
    if mask is not None:
        S = S.masked_fill(mask == 0, float('-inf'))
    
    # 步骤 3-5: 计算 Softmax
    # P: [batch, seq_len, seq_len]
    P = F.softmax(S, dim=-1)
    
    # 步骤 6-8: 计算输出 O = PV
    # P: [batch, seq_len, seq_len]
    # V: [batch, seq_len, dim]
    # O: [batch, seq_len, dim]
    O = torch.matmul(P, V)
    
    return O, P


def multi_head_attention(Q, K, V, num_heads, mask=None):
    """
    多头注意力实现
    
    参数:
        Q, K, V: [batch, seq_len, d_model]
        num_heads: 注意力头数
        mask: 可选的掩码
    
    返回:
        output: [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = Q.shape
    assert d_model % num_heads == 0
    
    d_k = d_model // num_heads
    
    # 重塑为多头格式
    # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
    # -> [batch, num_heads, seq_len, d_k]
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    # 计算注意力
    # [batch, num_heads, seq_len, d_k]
    O, P = naive_attention(Q, K, V, mask)
    
    # 合并多头
    # [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k]
    # -> [batch, seq_len, d_model]
    O = O.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    return O


# 示例使用
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_len = 512
    d_model = 512
    num_heads = 8
    
    # 创建输入
    Q = torch.randn(batch_size, seq_len, d_model, device='cuda')
    K = torch.randn(batch_size, seq_len, d_model, device='cuda')
    V = torch.randn(batch_size, seq_len, d_model, device='cuda')
    
    # 单头注意力
    O, P = naive_attention(Q, K, V)
    print(f"输出形状: {O.shape}")  # [2, 512, 512]
    print(f"注意力权重形状: {P.shape}")  # [2, 512, 512]
    
    # 多头注意力
    O_multi = multi_head_attention(Q, K, V, num_heads)
    print(f"多头输出形状: {O_multi.shape}")  # [2, 512, 512]
    
    # 内存占用分析
    S_size = batch_size * seq_len * seq_len * 4  # float32
    print(f"\n中间矩阵 S 的内存占用: {S_size / 1024 / 1024:.2f} MB")
```

### 6.2 NumPy 实现（更底层）

```python
import numpy as np

def softmax(x, axis=-1):
    """数值稳定的 Softmax"""
    # 减去最大值（数值稳定性）
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_step_by_step(Q, K, V):
    """
    逐步展示 Attention 计算过程
    
    参数:
        Q, K, V: [seq_len, dim] 的 numpy 数组
    
    返回:
        O: 输出 [seq_len, dim]
    """
    seq_len, dim = Q.shape
    
    print("=" * 60)
    print("原始 Attention 逐步计算")
    print("=" * 60)
    
    # 步骤 1: Q, K 已在内存中
    print(f"\n步骤 1: Q 形状 {Q.shape}, K 形状 {K.shape}")
    
    # 步骤 2: 计算 S = QK^T
    print(f"\n步骤 2: 计算 S = QK^T")
    S = np.matmul(Q, K.T)
    print(f"  S 形状: {S.shape}")
    print(f"  S 内存占用: {S.nbytes / 1024:.2f} KB")
    print(f"  S 示例值:\n{S[:3, :3]}")
    
    # 缩放
    scale = np.sqrt(dim)
    S = S / scale
    print(f"  缩放因子: {scale:.2f}")
    
    # 步骤 3-4: S 写回再读取（这里是概念性的）
    print(f"\n步骤 3-4: S 写回 HBM 再读取（模拟）")
    
    # 步骤 5: 计算 Softmax
    print(f"\n步骤 5: 计算 P = Softmax(S)")
    P = softmax(S, axis=-1)
    print(f"  P 形状: {P.shape}")
    print(f"  P 内存占用: {P.nbytes / 1024:.2f} KB")
    print(f"  P 示例值（第一行）:\n{P[0, :5]}")
    print(f"  P 每行求和（应该为1）: {P[0].sum():.6f}")
    
    # 步骤 6-7: P 写回再读取
    print(f"\n步骤 6-7: P 写回 HBM 再读取（模拟）")
    
    # 步骤 8: 计算 O = PV
    print(f"\n步骤 8: 计算 O = PV")
    O = np.matmul(P, V)
    print(f"  O 形状: {O.shape}")
    print(f"  O 内存占用: {O.nbytes / 1024:.2f} KB")
    print(f"  O 示例值:\n{O[:3, :3]}")
    
    # 步骤 9-10: 返回结果
    print(f"\n步骤 9-10: 写回结果并返回")
    
    # 总结
    print("\n" + "=" * 60)
    print("内存占用总结")
    print("=" * 60)
    total_intermediate = S.nbytes + P.nbytes
    print(f"输入 (Q, K, V): {(Q.nbytes + K.nbytes + V.nbytes) / 1024:.2f} KB")
    print(f"中间矩阵 (S, P): {total_intermediate / 1024:.2f} KB")
    print(f"输出 (O): {O.nbytes / 1024:.2f} KB")
    print(f"总计: {(Q.nbytes + K.nbytes + V.nbytes + total_intermediate + O.nbytes) / 1024:.2f} KB")
    
    return O, S, P


# 示例
if __name__ == "__main__":
    np.random.seed(42)
    
    # 小规模示例
    seq_len = 8
    dim = 4
    
    Q = np.random.randn(seq_len, dim).astype(np.float32)
    K = np.random.randn(seq_len, dim).astype(np.float32)
    V = np.random.randn(seq_len, dim).astype(np.float32)
    
    O, S, P = attention_step_by_step(Q, K, V)
    
    # 大规模示例（内存分析）
    print("\n\n" + "=" * 60)
    print("大规模场景分析（Llama3-8B）")
    print("=" * 60)
    
    N = 8192
    d = 128
    num_heads = 32
    
    S_size = N * N * 4  # float32
    total_per_head = S_size * 2  # S + P
    total_all_heads = total_per_head * num_heads
    
    print(f"序列长度 N: {N}")
    print(f"特征维度 d: {d}")
    print(f"注意力头数: {num_heads}")
    print(f"\n单头中间矩阵 (S + P): {total_per_head / 1024 / 1024:.2f} MB")
    print(f"所有头中间矩阵: {total_all_heads / 1024 / 1024:.2f} MB")
    print(f"所有头中间矩阵: {total_all_heads / 1024 / 1024 / 1024:.2f} GB")
```

### 6.3 CUDA 伪代码（概念层面）

```cuda
// 原始 Attention 的 CUDA 实现（简化版）

__global__ void attention_kernel(
    float* Q,      // [N, d]
    float* K,      // [N, d]
    float* V,      // [N, d]
    float* O,      // [N, d]
    int N,
    int d
) {
    // 步骤 1-2: 计算 S = QK^T
    // 需要多个 kernel 或者分块处理
    
    // 步骤 3: 写回 S 到全局内存
    // __global__ float* S;  // [N, N] 存储在 HBM
    
    // 步骤 4-5: 计算 Softmax
    // 每个线程处理一行
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        // 找最大值
        float max_val = -INFINITY;
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, S[row * N + i]);
        }
        
        // 计算 exp 和 sum
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float exp_val = expf(S[row * N + i] - max_val);
            P[row * N + i] = exp_val;
            sum += exp_val;
        }
        
        // 归一化
        for (int i = 0; i < N; i++) {
            P[row * N + i] /= sum;
        }
    }
    
    // 步骤 6: 写回 P 到全局内存
    
    // 步骤 7-8: 计算 O = PV
    // 需要另一个 kernel
    
    // 步骤 9: 写回 O
}

// 问题：
// 1. S 和 P 都需要存储在全局内存（慢）
// 2. 多次 kernel 调用，每次都要访问全局内存
// 3. 无法充分利用 shared memory
```

---

## 七、实际案例分析（Llama3-8B）

### 模型参数

| 参数 | 值 |
|------|-----|
| 序列长度 (N) | 8192 |
| 隐藏层维度 (d_model) | 4096 |
| 注意力头数 (num_heads) | 32 |
| 每头维度 (d_k) | 128 |
| 层数 | 32 |

### 单层单头内存分析

```
输入:
  Q: [8192, 128] = 1,048,576 个 float32 = 4 MB
  K: [8192, 128] = 1,048,576 个 float32 = 4 MB
  V: [8192, 128] = 1,048,576 个 float32 = 4 MB

中间矩阵:
  S: [8192, 8192] = 67,108,864 个 float32 = 268 MB
  P: [8192, 8192] = 67,108,864 个 float32 = 268 MB

输出:
  O: [8192, 128] = 1,048,576 个 float32 = 4 MB

总计（单头）: 552 MB
```

### 单层所有头内存分析

```
32 个头 × 268 MB (S + P) = 8,576 MB ≈ 8.6 GB
```

### 全模型内存分析

```
32 层 × 8.6 GB = 275 GB
```

**结论**：仅存储中间的注意力矩阵就需要 **275 GB** 显存！这是不可接受的。

### 实际测试（PyTorch）

```python
import torch
import time

def benchmark_attention(seq_len, dim, device='cuda'):
    """测试 Attention 的时间和内存"""
    Q = torch.randn(1, seq_len, dim, device=device)
    K = torch.randn(1, seq_len, dim, device=device)
    V = torch.randn(1, seq_len, dim, device=device)
    
    # 预热
    for _ in range(10):
        O = torch.matmul(torch.matmul(Q, K.transpose(-2, -1)), V)
    
    torch.cuda.synchronize()
    
    # 测试时间
    start = time.time()
    for _ in range(100):
        S = torch.matmul(Q, K.transpose(-2, -1))
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000  # ms
    
    # 内存占用
    S_memory = seq_len * seq_len * 4 / 1024 / 1024  # MB
    
    return avg_time, S_memory

# 测试不同序列长度
print("序列长度 | 时间 (ms) | 中间矩阵内存 (MB)")
print("-" * 50)
for N in [512, 1024, 2048, 4096, 8192]:
    try:
        time_ms, mem_mb = benchmark_attention(N, 128)
        print(f"{N:8d} | {time_ms:9.2f} | {mem_mb:18.2f}")
    except RuntimeError as e:
        print(f"{N:8d} | OOM (显存不足)")
```

---

## 八、优化方向

### 8.1 FlashAttention 的改进

FlashAttention 通过以下技术解决原始 Attention 的问题：

#### 1. **分块计算（Tiling）**
- 将 Q, K, V 分成小块
- 每次只在 SRAM 中处理一小块
- 避免存储完整的 $N \times N$ 矩阵

#### 2. **融合操作（Kernel Fusion）**
- 将 $QK^T \rightarrow \text{Softmax} \rightarrow PV$ 融合在一个 kernel 中
- 中间结果不写回 HBM

#### 3. **在线 Softmax（Online Softmax）**
- 增量式计算 Softmax，不需要完整的 S 矩阵
- 只需要维护当前的最大值和累加和

#### 4. **重计算（Recomputation）**
- 反向传播时重新计算某些值
- 用计算换内存

### 8.2 性能对比

| 指标 | 原始 Attention | FlashAttention |
|------|---------------|----------------|
| 内存复杂度 | $O(N^2)$ | $O(N)$ |
| IO 复杂度 | $O(N^2)$ | $O(N)$ |
| HBM 访问 | $2N^2 + 4Nd$ | $O(Nd)$ |
| 速度提升 | 1x | 2-4x |
| 支持序列长度 | 受限 | 更长 |

### 8.3 其他优化技术

1. **稀疏注意力**
   - 只计算部分注意力权重
   - 例如：局部窗口、步长注意力

2. **低秩近似**
   - 用低秩矩阵近似 $N \times N$ 的注意力矩阵
   - 例如：Linformer, Performer

3. **量化**
   - 使用 int8 或 fp16 代替 fp32
   - 减少内存占用和传输时间

4. **多查询注意力（MQA）/ 分组查询注意力（GQA）**
   - 多个 Query 共享 Key 和 Value
   - 减少 KV cache 大小

---

## 九、总结

### 原始 Attention 的核心问题

1. **$O(N^2)$ 的空间复杂度**：中间矩阵 S 和 P 占用大量内存
2. **频繁的 HBM 访问**：S 和 P 需要写回再读取
3. **IO 瓶颈**：大部分时间浪费在数据传输上
4. **无法处理长序列**：序列长度翻倍，内存占用增加 4 倍

### 关键洞察

> **计算不是瓶颈，内存访问才是瓶颈！**

现代 GPU 的计算能力（FLOPS）远超内存带宽，优化的关键在于：
- 减少 HBM 访问次数
- 充分利用快速的 SRAM（shared memory）
- 设计 IO-aware 的算法

### 学习路径

1. **理解原始 Attention**（本文档）
2. **学习 CUDA 编程基础**
3. **研究 FlashAttention 论文**
4. **实现简化版的 FlashAttention**
5. **优化到接近官方性能**

---

## 参考资料

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

### 代码
- [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)
- [PyTorch Attention 实现](https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py)

### 教程
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS 文档](https://github.com/NVIDIA/cutlass)

---

**文档创建时间**: 2025-11-16  
**作者**: AI Assistant  
**版本**: 1.0

