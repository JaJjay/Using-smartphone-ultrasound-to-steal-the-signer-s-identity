import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data = data.astype(float)
    y = lfilter(b, a, data)
    return y

def sliding_window_detection(signal, window_size=10, threshold=0.1):
    """
    创新的滑动窗口信号检测算法。
    算法思想：通过滑动窗口计算窗口内的变化幅度，如果窗口内变化幅度超过阈值，则将窗口的中心点标记为检测点。
    """
    detections = []
    half_window = window_size // 10
    for i in range(half_window, len(signal) - half_window):
        window_start = i - half_window
        window_end = i + half_window
        window_diff = np.max(signal[window_start:window_end]) - np.min(signal[window_start:window_end])
        if window_diff > threshold:
            detections.append(i)  # 标记为动作开始的点
    return detections

# 读取CSV文件
csv_file_path = 'yth2.csv'
df = pd.read_csv(csv_file_path)

# 假设采样频率fs和低通滤波的截止频率cutoff已知
fs = 20000.0  # 采样频率
cutoff = 2000.0  # 截止频率

# 对每个信号列应用滤波
filtered_signals = {}
for column in df.columns[1:]:  # 跳过时间标签列
    signal = df[column].values
    filtered_signal = butter_lowpass_filter(signal, cutoff, fs, order=5)
    filtered_signals[column] = filtered_signal

# 应用信号检测算法
detections = {}
for column, filtered_signal in filtered_signals.items():
    detected_points = sliding_window_detection(filtered_signal)
    detections[column] = detected_points

# 打印检测到的点（示例）
for column, points in detections.items():
    print(f"Detected points in {column}: {points}")

# 可视化一个信号及其检测点（仅作为示例）
plt.figure(figsize=(10, 6))
sample_signal = list(filtered_signals.keys())[5]  # 选择第一个信号列进行示例
plt.plot(filtered_signals[sample_signal], label='Filtered Signal')
for point in detections[sample_signal]:
    plt.plot(point, filtered_signals[sample_signal][point], 'ro')  # 标记检测点
plt.title(f"Signal Detection in '{sample_signal}'")
plt.legend()
plt.show()
