import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import re

# 加载特征文件
person1_feature_data = {
    '1850': [[1,2,3,4,3,2,5,3,4,2,3,4],[1,2,3,4,3,2,5,3,4,2,1,4],[1,2,4,4,2,2,5,3,4,2,3,4],[1,2,3,2,3,2,5,5,4,2,2,4]],
    '1851': [[1,2,3,4,3,2,5,3,4,2,3,4],[1,2,3,4,3,2,5,3,4,2,1,4],[1,2,4,4,2,2,5,3,4,2,3,4],[1,2,3,2,3,2,5,5,4,2,2,4]],
    '1852': [[1,2,3,4,3,2,5,3,4,2,3,4],[1,2,3,4,3,2,5,3,4,2,1,4],[1,2,4,4,2,2,5,3,4,2,3,4],[1,2,3,2,3,2,5,5,4,2,2,4]]
}

person2_feature_data = {
    '1850': [[14,13,12,15,13,14,12,13,14],[11,12,13,14,13,12,15,13,14,12,11,14],[11,12,14,14,12,12,15,13,14,12,13,14],[11,12,13,12,13,12,15,15,14,12,12,14]],
    '1851': [[11,12,13,14,13,12,15,13,14,12,13,14],[11,12,13,14,13,12,15,13,14,12,11,14],[11,12,14,14,12,12,15,13,14,12,13,14],[11,12,13,12,13,12,15,15,14,12,12,14]],
    '1852': [[11,12,13,14,13,12,15,13,14,12,13,14],[11,12,13,14,13,12,15,13,14,12,11,14],[11,12,14,14,12,12,15,13,14,12,13,14],[11,12,13,12,13,12,15,15,14,12,12,14]]
}

person3_feature_data = {
    '1850': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
    '1851': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
    '1852': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]]
}

# 构建 DataFrame
dfs = []
for person_data in [person1_feature_data, person2_feature_data, person3_feature_data]:
    for person, data in person_data.items():
        person_df = pd.DataFrame(data)
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
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

# 在测试集上评估模型
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)





# 提取每一轮的训练和验证loss
train_loss = []
val_loss = []

for line in history.history['loss']:
    train_loss.append(float(re.findall(r'loss: ([\d.]+)', line)[0]))

for line in history.history['val_loss']:
    val_loss.append(float(re.findall(r'val_loss: ([\d.]+)', line)[0]))

# 画出每一轮的loss曲线
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

