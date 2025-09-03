import os
import numpy as np
import pandas as pd
import re
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def collect_labels(data_path, labels_path, window_size, step_size):
    """收集所有窗口的标签"""
    labels = []
    group_names = ["grouped_1", "grouped_2"]

    for group_name in group_names:
        group_path = os.path.join(data_path, group_name)
        label_num = group_name.split("_")[-1]  # 提取组名末尾数字
        label_file = os.path.join(labels_path, f"{label_num}.csv")  # 动态匹配标签文件

        if not os.path.exists(group_path):
            print(f"警告：组目录不存在 {group_path}")
            continue

        try:
            labels_df = pd.read_csv(label_file, header=None)
            trial_labels = labels_df.values.flatten()
        except Exception as e:
            print(f"加载标签文件 {label_file} 失败: {str(e)}")
            continue

        subjects = [str(i) for i in range(1, 16)]
        for subject in subjects:
            subject_path = os.path.join(group_path, subject)
            if not os.path.exists(subject_path):
                print(f"警告：被试者目录不存在 {subject_path}")
                continue

            trial_files = sorted([f for f in os.listdir(subject_path) if f.endswith('.csv')], key=natural_sort_key)
            for trial_idx, trial_file in enumerate(trial_files):
                trial_path = os.path.join(subject_path, trial_file)
                try:
                    data_df = pd.read_csv(trial_path)
                    data_df.columns = data_df.columns.str.strip()
                    trial_values = data_df.values

                    start = 0
                    while start + window_size <= trial_values.shape[0]:
                        labels.append(trial_labels[trial_idx])
                        start += step_size
                except Exception as e:
                    print(f"处理文件 {trial_path} 失败: {str(e)}")
                    continue

    return np.array(labels, dtype=int)

def process_and_split(data_path, labels_path, channel, window_size, step_size, train_idx, test_idx):
    """处理数据并按索引分割训练集和测试集（优化内存管理）"""
    with h5py.File(os.path.join(data_path, "train_data.h5"), 'w') as h5_train, \
            h5py.File(os.path.join(data_path, "test_data.h5"), 'w') as h5_test:

        # 训练集数据集
        train_X = h5_train.create_dataset(
            'X',
            shape=(0, channel, window_size),
            maxshape=(None, channel, window_size),
            dtype=np.float32,
            chunks=(1, channel, window_size),
            compression="gzip"
        )
        train_Y = h5_train.create_dataset(
            'Y',
            shape=(0,),
            maxshape=(None,),
            dtype=int,
            chunks=(1,),
            compression="gzip"
        )

        # 测试集数据集
        test_X = h5_test.create_dataset(
            'X',
            shape=(0, channel, window_size),
            maxshape=(None, channel, window_size),
            dtype=np.float32,
            chunks=(1, channel, window_size),
            compression="gzip"
        )
        test_Y = h5_test.create_dataset(
            'Y',
            shape=(0,),
            maxshape=(None,),
            dtype=int,
            chunks=(1,),
            compression="gzip"
        )

        current_idx = 0  # 全局样本索引
        scaler = StandardScaler()  # 数据标准化

        group_names = ["grouped_1", "grouped_2"]

        print("\n" + "=" * 40)
        print("开始处理数据文件：")

        for group_name in group_names:
            group_path = os.path.join(data_path, group_name)
            label_num = group_name.split("_")[-1]  # 提取组名末尾数字
            label_file = os.path.join(labels_path, f"{label_num}.csv")  # 动态匹配标签文件

            if not os.path.exists(group_path):
                print(f"警告：组目录不存在 {group_path}")
                continue

            try:
                trial_labels = pd.read_csv(label_file, header=None).values.flatten()
            except Exception as e:
                print(f"加载标签文件 {label_file} 失败: {str(e)}")
                continue

            subjects = [str(i) for i in range(1, 16)]
            for subject in subjects:
                subject_dir = os.path.join(group_path, subject)
                if not os.path.exists(subject_dir):
                    print(f"警告：被试者目录不存在 {subject_dir}")
                    continue

                trial_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.csv')], key=natural_sort_key)
                for trial_idx, trial_file in enumerate(trial_files):
                    print(f"正在处理：{os.path.join(group_name, subject, trial_file)}")
                    trial_path = os.path.join(subject_dir, trial_file)

                    try:
                        buffer = np.empty((0, channel), dtype=np.float32)
                        chunk_size = 5000  # 减小分块大小以降低内存压力

                        for chunk in pd.read_csv(trial_path, chunksize=chunk_size):
                            chunk.columns = chunk.columns.str.strip()
                            chunk_data = chunk.iloc[:, channel_indices].values.astype(np.float32)
                            chunk_data = scaler.fit_transform(chunk_data)  # 标准化

                            combined_data = np.vstack([buffer, chunk_data])
                            start = 0

                            while start + window_size <= combined_data.shape[0]:
                                window = combined_data[start:start+window_size, :].T  # (channels, window)
                                label = trial_labels[trial_idx]

                                if current_idx in train_idx:
                                    train_X.resize(train_X.shape[0]+1, axis=0)
                                    train_X[-1] = window
                                    train_Y.resize(train_Y.shape[0]+1, axis=0)
                                    train_Y[-1] = label
                                elif current_idx in test_idx:
                                    test_X.resize(test_X.shape[0]+1, axis=0)
                                    test_X[-1] = window
                                    test_Y.resize(test_Y.shape[0]+1, axis=0)
                                    test_Y[-1] = label

                                current_idx += 1
                                start += step_size

                            buffer = combined_data[- (window_size - 1):, :] if combined_data.shape[0] >= (window_size - 1) else combined_data

                    except Exception as e:
                        print(f"读取文件 {trial_path} 失败: {str(e)}")
                        continue

# 标准10-20系统电极配置
selected_channels = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'C3', 'CZ', 'C4',
    'P3', 'PZ', 'P4', 'O1', 'O2'
]

channel_indices = [
    0, 2, 5, 7, 9, 11, 13,  # 前额叶和额叶
    25, 27, 29,  # 中央区
    43, 45, 47,  # 顶叶
    58,  60  # 枕叶
]
num_selected_channels = len(channel_indices)  # 15通道

# 参数设置
data_path = r"C:\Users\yijing\Desktop\DengYi\EEG\datasets\use_data"
labels_path = os.path.join(data_path, "labels")
window_size = 256
step_size = 32

# 收集标签并分割数据集
all_labels = collect_labels(data_path, labels_path, window_size, step_size)
indices = np.arange(len(all_labels))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

# 处理数据
process_and_split(
    data_path,
    labels_path,
    num_selected_channels,
    window_size,
    step_size,
    set(train_indices),
    set(test_indices)
)

print("处理完成！训练集和测试集已保存为HDF5文件。")