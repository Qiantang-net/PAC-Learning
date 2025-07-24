import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from kmeans_pytorch import kmeans
import time



def add_random_mask(x, window_sizes, mask_ratio=0.1):
    """
    根据不同大小的窗口随机掩码时间序列中的时间点。

    Args:
        x (torch.Tensor): 输入的时间序列，形状为 (B, T, C)
        window_sizes (list of int): 窗口大小的列表，每个窗口大小对应一个掩码操作
        mask_ratio (float): 掩码比例，控制每个窗口中多少比例的时间步会被掩码

    Returns:
        torch.Tensor: 添加掩码后的时间序列
    """
    """
        对时间序列数据进行随机掩码。

        参数:
        - data (torch.Tensor): 输入时间序列数据，形状为 (B, T, C)。
        - mask_ratio (float): 掩码率，表示需要掩码的时间点比例 (0.0 - 1.0)。

        返回:
        - masked_data (torch.Tensor): 掩码后的数据。
        - mask (torch.Tensor): 掩码矩阵，1 表示掩码位置，0 表示未掩码位置。
        """
    # 获取数据的形状
    data = x
    B, T, C = data.shape

    # 创建掩码矩阵，形状为 (B, T)
    mask = torch.rand(B, T) < mask_ratio  # 随机生成掩码位置

    # 在掩码位置，将数据置为零 (或其他掩码值)
    masked_data = data.clone()  # 复制数据
    masked_data[mask] = 0  # 掩码位置的数据置为 0
    return masked_data


def add_gaussian_noise(x, noise_ratio=0.1, noise_std=0.01):
    """
    在输入的时间序列 x 中添加高斯噪声。

    Args:
        x (torch.Tensor): 输入的时间序列，形状为 (B, T, C)
        noise_ratio (float): 噪声比例，控制在多少比例的时间步中加入噪声
        noise_std (float): 高斯噪声的标准差

    Returns:
        torch.Tensor: 添加噪声后的时间序列
    """
    B, T, C = x.shape
    num_noise_points = int(noise_ratio * T)  # 计算噪声点的数量

    # 随机选择要插入噪声的时间步
    noise_indices = torch.rand(B, num_noise_points, device=x.device).mul(T).long()  # 随机生成噪声点的索引

    # 对每个批次添加噪声
    noisy_x = x.clone()
    for b in range(B):
        for idx in noise_indices[b]:
            # 为每个选择的时间步生成高斯噪声
            noise = torch.randn(C, device=x.device) * noise_std
            noisy_x[b, idx] += noise  # 将噪声添加到时间序列中

    return noisy_x


def add_gamma_noise(x, noise_ratio=0.1, concentration=2.0, rate=1.0):
    """
    在输入的时间序列 x 中添加伽马噪声。

    Args:
        x (torch.Tensor): 输入的时间序列，形状为 (B, T, C)
        noise_ratio (float): 噪声比例，控制在多少比例的时间步中加入噪声
        concentration (float): 伽马分布的形状参数（k）
        rate (float): 伽马分布的速率参数（theta）

    Returns:
        torch.Tensor: 添加伽马噪声后的时间序列
    """
    B, T, C = x.shape
    num_noise_points = int(noise_ratio * T)  # 计算噪声点的数量

    # 随机选择要插入噪声的时间步
    noise_indices = torch.rand(B, num_noise_points, device=x.device).mul(T).long()

    # 对每个批次添加噪声
    noisy_x = x.clone()
    gamma_dist = torch.distributions.Gamma(concentration, rate)
    for b in range(B):
        for idx in noise_indices[b]:
            noise = gamma_dist.sample((C,)).to(x.device)  # 生成伽马噪声
            noisy_x[b, idx] += noise  # 将噪声添加到时间序列中

    return noisy_x


def add_rayleigh_noise(x, noise_ratio=0.1, scale=1.0):
    """
    在输入的时间序列 x 中添加瑞利噪声。

    Args:
        x (torch.Tensor): 输入的时间序列，形状为 (B, T, C)
        noise_ratio (float): 噪声比例，控制在多少比例的时间步中加入噪声
        scale (float): 瑞利分布的尺度参数

    Returns:
        torch.Tensor: 添加瑞利噪声后的时间序列
    """
    B, T, C = x.shape
    num_noise_points = int(noise_ratio * T)  # 计算噪声点的数量

    # 随机选择要插入噪声的时间步
    noise_indices = torch.rand(B, num_noise_points, device=x.device).mul(T).long()

    # 对每个批次添加噪声
    noisy_x = x.clone()
    weibull_dist = torch.distributions.Weibull(scale, 2.0)  # 形状参数 k=2
    for b in range(B):
        for idx in noise_indices[b]:
            noise = weibull_dist.sample((C,)).to(x.device)  # 生成瑞利噪声
            noisy_x[b, idx] += noise  # 将噪声添加到时间序列中

    return noisy_x


def add_exponential_noise(x, noise_ratio=0.1, rate=1.0):
    """
    在输入的时间序列 x 中添加指数噪声。

    Args:
        x (torch.Tensor): 输入的时间序列，形状为 (B, T, C)
        noise_ratio (float): 噪声比例，控制在多少比例的时间步中加入噪声
        rate (float): 指数分布的速率参数（lambda）

    Returns:
        torch.Tensor: 添加指数噪声后的时间序列
    """
    B, T, C = x.shape
    num_noise_points = int(noise_ratio * T)  # 计算噪声点的数量

    # 随机选择要插入噪声的时间步
    noise_indices = torch.rand(B, num_noise_points, device=x.device).mul(T).long()

    # 对每个批次添加噪声
    noisy_x = x.clone()
    exponential_dist = torch.distributions.Exponential(rate)
    for b in range(B):
        for idx in noise_indices[b]:
            noise = exponential_dist.sample((C,)).to(x.device)  # 生成指数噪声
            noisy_x[b, idx] += noise  # 将噪声添加到时间序列中

    return noisy_x


def add_uniform_noise(x, noise_ratio=0.1, low=-1.0, high=1.0):
    """
    在输入的时间序列 x 中添加均匀噪声。

    Args:
        x (torch.Tensor): 输入的时间序列，形状为 (B, T, C)
        noise_ratio (float): 噪声比例，控制在多少比例的时间步中加入噪声
        low (float): 均匀分布的下界
        high (float): 均匀分布的上界

    Returns:
        torch.Tensor: 添加均匀噪声后的时间序列
    """
    B, T, C = x.shape
    num_noise_points = int(noise_ratio * T)  # 计算噪声点的数量

    # 随机选择要插入噪声的时间步
    noise_indices = torch.rand(B, num_noise_points, device=x.device).mul(T).long()

    # 对每个批次添加噪声
    noisy_x = x.clone()
    for b in range(B):
        for idx in noise_indices[b]:
            noise = torch.rand(C, device=x.device) * (high - low) + low  # 生成均匀噪声
            noisy_x[b, idx] += noise  # 将噪声添加到时间序列中

    return noisy_x



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def k_means_clustering(x,n_mem,d_model):
    start = time.time()

    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    # sckit-learn xxxx (cuda problem)
    _, cluster_centers = kmeans(X=x, num_clusters=n_mem, distance='euclidean', device=torch.device('cuda:0'))
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers


def cut_add_paste(sample_xi, sample_xj, trend, cut_length, paste_position):
    """
    使用CutAddPaste生成伪异常样本。

    参数:
    - sample_xi: 原始时间序列样本i，形状为 (b,t, d)，其中 d 是维度数，t 是时间步长。
    - sample_xj: 时间序列样本j，形状为 (b, t, d)。
    - trend: 趋势信号，形状为 (b, cut_length, d)。
    - cut_length: 截取的时间段长度。
    - paste_position: 将伪异常片段粘贴到 sample_xi 的位置。

    返回:
    - 伪异常样本 Xa。
    """
    b, t, d = sample_xi.shape

    # 1. 从样本j中截取片段，确保长度为 cut_length
    cut_start = torch.randint(0, t - cut_length, (1,)).item()
    cut_segment = sample_xj[:, cut_start:cut_start + cut_length, :]  # 这里确保片段大小是 cut_length

    # 2. 添加趋势信号，确保趋势和 cut_segment 大小一致
    trend = trend.unsqueeze(0).repeat(b, 1, 1)  # 扩展到批量维度，shape: (b,cut_length,d)
    trend = trend.to('cuda')
    augmented_segment = cut_segment + trend  # 现在 cut_segment 和 trend 在时间维度都是 cut_length

    # 3. 将片段粘贴到样本i中
    pseudo_anomalous_sample = sample_xi.clone()
    pseudo_anomalous_sample[:, paste_position:paste_position + cut_length, :] = augmented_segment

    return pseudo_anomalous_sample

def adjust_amplitude_in_frequency_domain(x, factor):

    """
    This function converts the time series data `x` to the frequency domain,
    identifies the frequency with the largest amplitude, and adjusts its amplitude.

    :param x: Input tensor of shape (B, T, D), where B is the batch size,
              T is the time steps, and D is the number of features.
    :param factor: The factor by which to multiply the amplitude of the selected frequency.
    :return: Tensor with adjusted amplitudes in the time domain.
    """
    # Convert to frequency domain using FFT
    x_freq = torch.fft.fft(x, dim=1)  # FFT along time dimension (T)

    # Compute amplitude (magnitude) of each frequency
    amplitude = torch.abs(x_freq)  # Amplitude of each frequency component

    # Find the index of the maximum amplitude for each feature
    max_amplitude_index = torch.argmax(amplitude, dim=1, keepdim=True)  # Shape: (B, 1, D)

    # Adjust the amplitude of the frequency with the largest magnitude
    # Increase the amplitude of the maximum frequency by the factor
    for i in range(x.size(0)):  # Loop over the batch
        for j in range(x.size(2)):  # Loop over the features
            max_idx = max_amplitude_index[i, 0, j]
            x_freq[i, max_idx, j] *= factor  # Adjust the amplitude by multiplying the factor

    # Convert back to time domain using iFFT
    x_adjusted = torch.fft.ifft(x_freq, dim=1).real  # Take real part after iFFT

    return x_adjusted



# import time
# import torch
# from thop import profile  # 用于计算FLOPs
#
##########效率分析实验#########
# # 假设您已经定义了模型和数据加载器
# # 示例：model = YourModel(), train_loader, test_loader
#
# def measure_training_time(model, train_loader, optimizer, criterion, device, epochs=1):
#     """
#     测量模型每个epoch的训练时间。
#     """
#     model.to(device)
#     model.train()
#
#     start_time = time.time()
#     for epoch in range(epochs):
#         for batch in train_loader:
#             inputs, labels = batch
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # 前向传播
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     end_time = time.time()
#
#     avg_time_per_epoch = (end_time - start_time) / epochs
#     return avg_time_per_epoch
#
#
# def measure_testing_time(model, test_loader, device):
#     """
#     测量模型测试时间。
#     """
#     model.to(device)
#     model.eval()
#
#     start_time = time.time()
#     with torch.no_grad():
#         for batch in test_loader:
#             inputs, _ = batch
#             inputs = inputs.to(device)
#             _ = model(inputs)
#     end_time = time.time()
#
#     total_time = end_time - start_time
#     return total_time
#
#
# def measure_flops(model, input_size):
#     """
#     测量模型的FLOPs。
#     """
#     # 创建一个随机输入张量，模拟输入数据
#     dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)
#     flops, params = profile(model, inputs=(dummy_input,), verbose=False)
#     return flops / 1e9  # 返回GFLOPs
#
#
# # 示例模型和数据（根据实际任务替换）
# # from your_model import YourModel
# # from your_dataloader import get_loaders
# # model = YourModel()
# # train_loader, test_loader = get_loaders()
# # optimizer = torch.optim.Adam(model.parameters())
# # criterion = torch.nn.CrossEntropyLoss()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 输入尺寸，例如 (batch_size, num_channels, height, width) 或时间序列维度
# input_size = (1, 3, 224, 224)  # 根据实际任务替换
#
# # 测量训练时间
# # train_time = measure_training_time(model, train_loader, optimizer, criterion, device)
#
# # 测量测试时间
# # test_time = measure_testing_time(model, test_loader, device)
#
# # 测量FLOPs
# # flops = measure_flops(model, input_size)
#
# # 打印结果
# # print(f"Training time (s/epoch): {train_time:.2f}")
# # print(f"Testing time (s): {test_time:.2f}")
# # print(f"FLOPs (G): {flops:.2f}")
# pip install thop



#######鲁棒性实验#########
#######加噪声######








#####丢弃部分数据#####





#######掩码##########