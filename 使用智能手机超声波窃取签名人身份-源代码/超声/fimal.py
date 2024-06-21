import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import csv
import pywt
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data = data.astype(float)
    y = lfilter(b, a, data)
    return y

def normalize_coefficients(coeffs, min_val, max_val):
    """
    将系数归一化到指定范围内。
    """
    coeffs_abs_max = np.max(np.abs(coeffs))
    normalized_coeffs = (coeffs / coeffs_abs_max) * (max_val - min_val) + min_val
    return normalized_coeffs

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
csv_file_path = 'lhy.csv'
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

# 截取每次写名字时产生的超声波
sp = {}
for column, points in detections.items():
    sp_column = []
    p_start = 150
    p_range = []
    for i in range(0, len(points)-1):
        if points[i] > 150:
            if len(p_range) >= 1:
                p_range.append(points[i] - p_range[0] + 1)
            else:
                p_range.append(points[i])
        if points[i] > 150 and points[i+1] - p_start >= 75:
            if len(p_range) > 1:
                p_range[0] = 1
                sp_column.append(p_range)
            p_start = points[i+1]
            p_range = []
    sp[column] = sp_column

# 打印检测到的点（示例）
for column, points in detections.items():
    print(f"Detected points in {column}: {points}")

for key in sp:
    print(f"s_points in {key}: {sp[key]}")

lll = {}
for key in sp:
    llll = []
    for list1 in sp[key]:
        cA, cD = pywt.dwt(list1, 'db1')

        # 归一化系数
        min_val = 1
        max_val = 100
        normalized_cA = normalize_coefficients(cA, min_val, max_val)
        normalized_cD = normalize_coefficients(cD, min_val, max_val)
        new_data =  normalized_cA + normalized_cD
        llll.append(list(new_data))
    lll[key] = llll

for key in lll:
    for i, sublist in enumerate(lll[key]):
        # 首先对每个元素加100
        updated_sublist = [x + 100 for x in sublist]
        # 再次归一化到0到100的范围
        normalized_sublist = normalize_coefficients(np.array(updated_sublist), 0, 100)
        lll[key][i] = list(normalized_sublist)

# 打开文件并写入字典内容
with open("lhy.txt", "w") as file:
    for key, value in lll.items():
        file.write(f"'{key}': {value}\n")
