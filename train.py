import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import h5py
import warnings
import csv
import random
import EEG.model
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

# 一、数据处理以及参数设置
# 参数设置
data_path = r"C:\Users\yijing\Desktop\DengYi\EEG\datasets\use_data"
max_length = 256  # 数据长度

# # 加载训练集
# print("start to load training data")
# with h5py.File(os.path.join(data_path, "train_data.h5"), 'r') as file:
#     X = np.array(file['X'])  # 形状 (N, 18, 256)
#     Y = np.array(file['Y'])  # 形状 (N,)
#
# # 加载测试集
# print("start to load testing data")
# with h5py.File(os.path.join(data_path, "test_data.h5"), 'r') as file:
#     M = np.array(file['X'])
#     L = np.array(file['Y'])
#
# # 转换为PyTorch张量并调整形状
# x_train = torch.tensor(X, dtype=torch.float32)
# x_train = x_train.permute(0, 2, 1).unsqueeze(1).permute(0, 1, 3, 2)  # [N, 1, 18, 256]
# y_train = torch.tensor(Y, dtype=torch.long)
#
# x_test = torch.tensor(M, dtype=torch.float32)
# x_test = x_test.permute(0, 2, 1).unsqueeze(1).permute(0, 1, 3, 2)  # [N, 1, 18, 256]
# y_test = torch.tensor(L, dtype=torch.long)


# 自定义HDF5数据集类（支持包大小和发送频率模拟）
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, scaler=None, fit_scaler=False, mode='train',
                 target_loss_ratio=0.0, packet_size=30, send_frequency=1.0):
        # 确保文件存在
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5文件不存在: {h5_path}")

        # 使用内存映射方式打开HDF5文件
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r', libver='latest', swmr=True)

        if 'X' not in self.h5_file or 'Y' not in self.h5_file:
            raise KeyError(f"HDF5文件中缺少'X'或'Y'数据集")

        # 获取数据集引用，不加载数据
        self.X_dataset = self.h5_file['X']
        self.Y_dataset = self.h5_file['Y']

        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.num_channels = self.X_dataset.shape[1]  # 动态获取通道数
        self.mode = mode  # 模式标识（train/test）

        # 传输参数
        self.target_loss_ratio = target_loss_ratio
        self.packet_size = packet_size
        self.send_frequency = send_frequency

        # 自动计算合适的drop_prob
        self.drop_prob = self.calculate_drop_prob()

        # 统计变量
        self.total_points = 0
        self.lost_points = 0
        self.total_packets = 0
        self.lost_packets = 0

        # 训练标准化器
        if self.fit_scaler and scaler is None:
            self.scaler = StandardScaler()
            chunk_size = 1000
            num_samples = len(self)

            # 分块训练标准化器
            for i in tqdm(range(0, num_samples, chunk_size), desc="训练标准化器"):
                end_idx = min(i + chunk_size, num_samples)
                chunk = self.X_dataset[i:end_idx].astype(np.float32)
                self.scaler.partial_fit(chunk.reshape(-1, self.num_channels))

    def calculate_drop_prob(self):
        """基于目标丢失比例和包大小计算合适的drop_prob"""
        # 每个样本的点数 = 通道数 × 时间步数
        points_per_sample = self.num_channels * self.X_dataset.shape[2]

        # 每个样本的包数量
        packets_per_sample = points_per_sample / self.packet_size

        # 计算理论drop_prob
        drop_prob = self.target_loss_ratio / (self.send_frequency * packets_per_sample)

        return min(0.5, max(0.001, drop_prob))

    def __len__(self):
        return self.X_dataset.shape[0]

    def __getitem__(self, idx):
        # 只加载当前需要的单个样本
        x = self.X_dataset[idx].astype(np.float32)  # 形状 (channels, time_steps)
        time_steps = x.shape[1]

        # 重置统计 - 每次只统计当前样本
        sample_points = self.num_channels * time_steps
        sample_lost_points = 0
        sample_lost_packets = 0

        # 只在训练时应用数据丢失模拟（且不是控制组）
        if self.mode == 'train' and self.target_loss_ratio > 0:
            # 调试信息 - 显示关键参数
            if idx == 0:  # 只打印第一个样本的调试信息
                print(f"[DEBUG] 模拟参数: drop_prob={self.drop_prob:.6f}, "
                      f"packet_size={self.packet_size}, freq={self.send_frequency}")

            # 模拟发送频率：随机跳过部分数据包
            if np.random.rand() > self.send_frequency:
                # 跳过当前样本的所有数据包 - 不进行任何处理
                pass
            else:
                # 计算当前样本的包数量
                num_packets = max(1, int(sample_points / self.packet_size))

                # 随机决定是否发生数据包丢失
                if np.random.rand() < self.drop_prob:
                    # 随机选择丢失的起始点
                    max_start = max(0, time_steps - self.packet_size)
                    start_idx = np.random.randint(0, max_start) if max_start > 0 else 0

                    # 应用包丢失
                    end_idx = min(start_idx + self.packet_size, time_steps)
                    lost_points = self.num_channels * (end_idx - start_idx)

                    x[:, start_idx:end_idx] = 0

                    # 记录丢失点数
                    sample_lost_points = lost_points
                    sample_lost_packets = 1

                # 添加轻微噪声（模拟网络抖动）
                x += np.random.normal(0, 0.005, x.shape)

        # 更新全局统计
        self.total_points += sample_points
        self.lost_points += sample_lost_points
        self.total_packets += max(1, int(sample_points / self.packet_size))
        self.lost_packets += sample_lost_packets

        # 应用标准化
        if self.scaler:
            # 标准化器要求特征在列上，所以转置为 (time_steps, channels)
            x = self.scaler.transform(x.T).T  # 转置标准化再转回

        # 转换为PyTorch张量并调整形状
        x_tensor = torch.tensor(x, dtype=torch.float32)  # 形状 (channels, time_steps)

        # 调整形状以匹配模型输入 [1, 1, channels, time_steps]
        x_tensor = x_tensor.unsqueeze(0)

        y = self.Y_dataset[idx]
        return x_tensor, y

    def get_actual_loss_ratio(self):
        """获取实际丢失比例"""
        if self.total_points == 0:
            return 0.0
        return self.lost_points / self.total_points

    def get_packet_loss_ratio(self):
        """获取包丢失比例"""
        if self.total_packets == 0:
            return 0.0
        return self.lost_packets / self.total_packets

    def reset_stats(self):
        """重置统计"""
        self.total_points = 0
        self.lost_points = 0
        self.total_packets = 0
        self.lost_packets = 0

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()


def train_model(config, experiment_name):
    """训练模型并返回结果"""
    # 初始化数据集
    train_dataset = HDF5Dataset(
        os.path.join(data_path, "train_data.h5"),
        fit_scaler=True,
        mode="train",
        target_loss_ratio=0.01 if not config.get('is_control', False) else 0.0,  # 控制组无丢失
        packet_size=config['packet_size'],
        send_frequency=config['send_frequency']
    )

    # 打印关键参数用于调试
    print(f"训练参数: packet_size={config['packet_size']}, "
          f"send_frequency={config['send_frequency']}, "
          f"drop_prob={train_dataset.drop_prob:.6f}")

    test_dataset = HDF5Dataset(
        os.path.join(data_path, "test_data.h5"),
        scaler=train_dataset.scaler,
        mode="test"
    )

    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    model = EEG.model.CNN_LSTM_ATTENTION_Model(num_classes=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # 动态调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # 初始化变量
    num_epochs = 200
    best_test_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    loss_ratios = []
    packet_loss_ratios = []

    # 创建结果目录
    os.makedirs(experiment_name, exist_ok=True)
    csv_file = os.path.join(experiment_name, 'training_log.csv')

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy',
                         'Test Accuracy', 'Loss Ratio', 'Packet Loss Ratio'])

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0


        # 训练阶段
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        # 计算实际丢失比例
        actual_loss_ratio = train_dataset.get_actual_loss_ratio()
        packet_loss_ratio = train_dataset.get_packet_loss_ratio()
        # 添加详细调试信息
        if epoch == 0:  # 只在第一个epoch打印详细统计
            print(f"统计详情: total_points={train_dataset.total_points}, "
                  f"lost_points={train_dataset.lost_points}, "
                  f"total_packets={train_dataset.total_packets}, "
                  f"lost_packets={train_dataset.lost_packets}")
        loss_ratios.append(actual_loss_ratio)
        packet_loss_ratios.append(packet_loss_ratio)



        # 计算训练指标
        train_loss_avg = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_accuracy)

        # 测试阶段
        # 每 10 个 epoch 进行一次测试
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            model.eval()
            correct_test = total_test = 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    _, predicted = torch.max(outputs, 1)
                    total_test += y_batch.size(0)
                    correct_test += (predicted == y_batch).sum().item()
            test_accuracy = correct_test / total_test
        else:
            # 其余 epoch 复用上一次结果（或填 None / NaN）
            test_accuracy = test_accuracies[-1] if test_accuracies else 0.0

        test_accuracies.append(test_accuracy)
        # 只在真正测试时才 step scheduler
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            scheduler.step(test_accuracy)

        # 更新最佳准确率
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(experiment_name, 'best_model.pth'))

        # 记录日志
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_loss_avg:.4f}",
                f"{train_accuracy:.4f}",
                f"{test_accuracy:.4f}",
                f"{actual_loss_ratio:.6f}",
                f"{packet_loss_ratio:.6f}"
            ])

        #全局统计
        actual_loss_ratio = train_dataset.get_actual_loss_ratio()
        packet_loss_ratio = train_dataset.get_packet_loss_ratio()


        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, "
              f"Loss Ratio: {actual_loss_ratio * 100:.3f}%, "
              f"Packet Loss: {packet_loss_ratio * 100:.3f}%")

        #重置比例
        train_dataset.reset_stats()

    # 返回最终结果
    return {
        'config': config,
        'best_test_accuracy': best_test_accuracy,
        'final_test_accuracy': test_accuracies[-1],
        'avg_train_accuracy': np.mean(train_accuracies[-10:]),
        'loss_ratio': np.mean(loss_ratios[-10:]),
        'packet_loss_ratio': np.mean(packet_loss_ratios[-10:])
    }

def run_experiments():
    """运行所有实验配置"""
    # 实验配置矩阵 - 固定丢失率为1%
    experiments = []

    # 包大小和发送频率组合
    packet_sizes = [10, 30, 50]  # 不同包大小 (时间点数)
    send_frequencies = [0.5, 1.0, 2.0]  # 不同发送频率 (0.5=半速, 1.0=全速, 2.0=双倍速)

    # 创建所有实验组合
    for packet_size in packet_sizes:
        for freq in send_frequencies:
            experiments.append({
                'packet_size': packet_size,
                'send_frequency': freq
            })

    # 添加控制组 (无丢失)
    experiments.append({
        'packet_size': 0,
        'send_frequency': 1.0,
        'is_control': True
    })

    # 运行所有实验
    results = []
    for i, config in enumerate(experiments):
        # 创建实验名称
        if config.get('is_control', False):
            exp_name = f"control_group"
        else:
            exp_name = (f"packet_{config['packet_size']}_"
                        f"freq_{config['send_frequency']}")

        print(f"\n{'=' * 50}")
        print(f"开始实验 {i + 1}/{len(experiments)}: {exp_name}")
        print(f"配置: {config}")

        # 运行训练
        result = train_model(config, exp_name)
        results.append(result)

        # 保存当前结果
        result_df = pd.DataFrame(results)
        result_df.to_csv('experiment_results_summary.csv', index=False)

        print(f"实验完成! 最佳测试准确率: {result['best_test_accuracy']:.4f}")

    return results


def analyze_results(results):
    """分析并可视化实验结果"""
    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 提取配置参数
    df['packet_size'] = df['config'].apply(lambda x: x['packet_size'])
    df['send_frequency'] = df['config'].apply(lambda x: x['send_frequency'])
    df['is_control'] = df['config'].apply(lambda x: x.get('is_control', False))

    # 保存完整结果
    df.to_csv('full_experiment_results.csv', index=False)

    # 1. 包大小对准确率的影响 (固定发送频率)
    plt.figure(figsize=(12, 8))
    for freq in [0.5, 1.0, 2.0]:
        subset = df[(df['send_frequency'] == freq) &
                    (~df['is_control'])]

        if not subset.empty:
            plt.plot(subset['packet_size'], subset['best_test_accuracy'],
                     marker='o', linestyle='-',
                     label=f"Freq {freq}")

    # 添加控制组
    control = df[df['is_control']]
    if not control.empty:
        plt.axhline(y=control['best_test_accuracy'].values[0], color='r',
                    linestyle='--', label='Control (No Loss)')

    plt.xlabel('Packet Size (Time Points)')
    plt.ylabel('Test Accuracy')
    plt.title('Impact of Packet Size on Model Accuracy (1% Data Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig('packet_size_impact.png', dpi=300)

    # 2. 发送频率对准确率的影响 (固定包大小)
    plt.figure(figsize=(12, 8))
    for packet_size in [10, 30, 50]:
        subset = df[(df['packet_size'] == packet_size) &
                    (~df['is_control'])]

        if not subset.empty:
            plt.plot(subset['send_frequency'], subset['best_test_accuracy'],
                     marker='s', linestyle='-',
                     label=f"Size {packet_size}")

    # 添加控制组
    if not control.empty:
        plt.axhline(y=control['best_test_accuracy'].values[0], color='r',
                    linestyle='--', label='Control (No Loss)')

    plt.xlabel('Send Frequency (Relative)')
    plt.ylabel('Test Accuracy')
    plt.title('Impact of Send Frequency on Model Accuracy (1% Data Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig('send_frequency_impact.png', dpi=300)

    # 3. 热力图：包大小和发送频率的综合影响
    plt.figure(figsize=(10, 8))
    subset = df[~df['is_control']]

    if not subset.empty:
        pivot = subset.pivot_table(index='packet_size',
                                   columns='send_frequency',
                                   values='best_test_accuracy')

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                    cbar_kws={'label': 'Test Accuracy'})
        plt.title('Model Accuracy at 1% Data Loss')
        plt.xlabel('Send Frequency')
        plt.ylabel('Packet Size')
        plt.tight_layout()
        plt.savefig('heatmap_accuracy.png', dpi=300)

    # 4. 包丢失率与数据丢失率的关系
    plt.figure(figsize=(12, 8))
    for freq in [0.5, 1.0, 2.0]:
        for packet_size in [10, 30, 50]:
            subset = df[(df['send_frequency'] == freq) &
                        (df['packet_size'] == packet_size) &
                        (~df['is_control'])]

            if not subset.empty:
                # 提取实际丢失比例
                actual_loss = subset['loss_ratio'].values * 100
                packet_loss = subset['packet_loss_ratio'].values * 100

                plt.scatter(packet_loss, actual_loss, s=100,
                            label=f"Freq {freq}, Size {packet_size}")

    # 添加参考线
    plt.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Ideal')
    plt.xlabel('Packet Loss Rate (%)')
    plt.ylabel('Data Loss Rate (%)')
    plt.title('Relationship Between Packet Loss and Data Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('packet_vs_data_loss.png', dpi=300)

    return df


def plot_final_comparison(df):
    """绘制最终对比图表"""
    # 提取实验组数据（排除控制组）
    experiment_data = df[~df['is_control']]

    if experiment_data.empty:
        return

    # 创建对比图表
    plt.figure(figsize=(14, 10))

    # 1. 所有配置的准确率对比
    plt.subplot(2, 2, 1)
    experiment_data['config_name'] = experiment_data.apply(
        lambda x: f"Size={x['packet_size']}, Freq={x['send_frequency']}", axis=1)

    # 按准确率排序
    experiment_data = experiment_data.sort_values('best_test_accuracy', ascending=False)

    plt.bar(experiment_data['config_name'], experiment_data['best_test_accuracy'], color='skyblue')
    plt.axhline(y=df[df['is_control']]['best_test_accuracy'].values[0], color='r',
                linestyle='--', label='Control (No Loss)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Test Accuracy')
    plt.title('Model Accuracy by Configuration (1% Data Loss)')
    plt.legend()
    plt.grid(True, axis='y')

    # 2. 传输效率分析
    plt.subplot(2, 2, 2)
    experiment_data['transmission_efficiency'] = experiment_data['best_test_accuracy'] / \
                                                 (experiment_data['packet_size'] * experiment_data['send_frequency'])

    # 按传输效率排序
    experiment_data = experiment_data.sort_values('transmission_efficiency', ascending=False)

    plt.bar(experiment_data['config_name'], experiment_data['transmission_efficiency'], color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Transmission Efficiency')
    plt.title('Transmission Efficiency (Accuracy / (Size × Freq))')
    plt.grid(True, axis='y')

    # 3. 包大小与准确率关系
    plt.subplot(2, 2, 3)
    for freq in [0.5, 1.0, 2.0]:
        subset = experiment_data[experiment_data['send_frequency'] == freq]
        if not subset.empty:
            plt.plot(subset['packet_size'], subset['best_test_accuracy'],
                     marker='o', linestyle='-', label=f"Freq {freq}")

    plt.xlabel('Packet Size')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy by Packet Size')
    plt.legend()
    plt.grid(True)

    # 4. 发送频率与准确率关系
    plt.subplot(2, 2, 4)
    for size in [10, 30, 50]:
        subset = experiment_data[experiment_data['packet_size'] == size]
        if not subset.empty:
            plt.plot(subset['send_frequency'], subset['best_test_accuracy'],
                     marker='s', linestyle='-', label=f"Size {size}")

    plt.xlabel('Send Frequency')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy by Send Frequency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    # 运行所有实验
    results = run_experiments()

    # 分析结果
    df = analyze_results(results)

    # 绘制最终对比图表
    plot_final_comparison(df)

    # 打印最佳配置
    best_config = df.loc[df['best_test_accuracy'].idxmax()]
    print("\n" + "=" * 50)
    print("最佳配置:")
    print(f"包大小: {best_config['packet_size']}")
    print(f"发送频率: {best_config['send_frequency']}")
    print(f"测试准确率: {best_config['best_test_accuracy']:.4f}")
    print("=" * 50)