import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import random

def augment_data(feature_data, num_augmentations=10, max_variation=0.2):
    augmented_data = {}
    for key, values in feature_data.items():
        augmented_samples = []
        for sample in values:
            for _ in range(num_augmentations):
                augmented_sample = [x + random.uniform(-max_variation, max_variation) * x for x in sample]
                augmented_samples.append(augmented_sample)
        augmented_data[key] = augmented_samples
    return augmented_data

# 加载特征文件
person_data = {
    'lhy': 'lc.txt',
    'lc': 'lrj.txt',
    'zlf': 'zlf.txt',
    'lrj': 'yth2.txt',
    'yth': 'lhy.txt'
}

feature_data = {}
for person, file_path in person_data.items():
    person_feature_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                key = key.strip()
                value = eval(value.strip())
                person_feature_data[key] = value
    feature_data[person] = person_feature_data

# 数据增强
augmented_data = {}
for person, data in feature_data.items():
    augmented_data[person] = augment_data(data)

# 构建 DataFrame
dfs = []
for person, data in augmented_data.items():
    for person_key, samples in data.items():
        person_df = pd.DataFrame(samples)
        person_df['Person'] = person
        dfs.append(person_df)

feature_df = pd.concat(dfs, ignore_index=True)

# 分割特征和目标标签
X = feature_df.drop(columns=['Person'])  # 特征
y = feature_df['Person']  # 标签

# 对标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)

# 定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 定义早停机制
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 在测试集上评估模型
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

# 提取每一轮的训练和验证loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 画出每一轮的loss曲线
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# 提取分类报告中的准确率、Macro avg和Weighted avg数据
accuracy_data = accuracy
macro_avg_data = report['macro avg']
weighted_avg_data = report['weighted avg']

# 绘制柱状图
plt.figure(figsize=(8, 6))

# 横坐标和数据
labels = ['Accuracy', 'Macro avg', 'Weighted avg']
data = [accuracy_data, macro_avg_data['f1-score'], weighted_avg_data['f1-score']]

# 绘制柱状图
plt.bar(labels, data, color=['blue', 'orange', 'green'], width=0.3)

# 添加数据标签
for i in range(len(labels)):
    plt.text(i, data[i], f"{data[i]:.2f}", ha='center', va='bottom')

# 添加标题和标签
plt.title('Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')

# 显示图形
plt.show()
