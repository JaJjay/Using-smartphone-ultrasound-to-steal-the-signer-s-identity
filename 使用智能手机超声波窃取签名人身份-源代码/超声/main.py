import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Step 1: 巴特沃斯滤波器定义
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Step 2: 信号动作检测
def detect_actions(signal, threshold=0.01, min_interval=1000):
    actions = []
    start = None
    for i, value in enumerate(signal):
        if abs(value) > threshold and start is None:
            start = i
        elif abs(value) <= threshold and start is not None:
            if i - start > min_interval:
                actions.append((start, i))
            start = None
    return actions

# Step 3: 保存动作到CSV
def save_actions_to_csv(actions, filename='actions.csv'):
    columns = ['start', 'end', 'label']
    data = []
    for start, end in actions:
        data.append([start, end, 'SignatureAction'])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

# 示例信号生成 (模拟数据)
fs = 48000  # Sample rate, Hz
cutoff = 2000  # Desired cutoff frequency of the filter, Hz
np.random.seed(0)
t = np.linspace(0, 1.0, fs)
# 生成一个模拟信号：正弦波 + 噪声
signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, t.shape)

# 应用巴特沃斯低通滤波器
filtered_signal = butter_lowpass_filter(signal, cutoff, fs)

# 检测动作
actions = detect_actions(filtered_signal, threshold=0.2, min_interval=500)

# 保存动作到CSV
save_actions_to_csv(actions)

# 可视化结果
plt.figure(figsize=(15, 6))
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal', linewidth=2)
for start, end in actions:
    plt.axvline(x=t[start], color='r', linestyle='--')
    plt.axvline(x=t[end], color='g', linestyle='--')
plt.legend()
plt.title('Signal and Detected Actions')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.show()
