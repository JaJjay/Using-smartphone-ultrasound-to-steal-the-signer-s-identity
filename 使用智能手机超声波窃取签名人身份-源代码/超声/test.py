#传统算法
#已实现在训练模型之前使用均值来填充 NaN 值

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import random
import matplotlib.pyplot as plt

num_runs = 5
svm_macro_avg = []
svm_weighted_avg = []
svm_accuracies = []
rf_macro_avg = []
rf_weighted_avg = []
rf_accuracies = []
nb_macro_avg = []
nb_weighted_avg = []
nb_accuracies = []


def augment_data(feature_data, num_augmentations=5, max_variation=0.1):
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

#第一个人
yth_feature_data = {}
with open('zlf.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = key.strip()
            value = eval(value.strip())
            yth_feature_data[key] = value

#第二个人
#person2_feature_data = {
#    '1850': [[11,12,13,14,13,12,15,13,14,12,13,14],[11,12,13,14,13,12,15,13,14,12,11,14],[11,12,14,14,12,12,15,13,14,12,13,14],[11,12,13,12,13,12,15,15,14,12,12,14]],
#    '1851': [[11,12,13,14,13,12,15,13,14,12,13,14],[11,12,13,14,13,12,15,13,14,12,11,14],[11,12,14,14,12,12,15,13,14,12,13,14],[11,12,13,12,13,12,15,15,14,12,12,14]],
#    '1852': [[11,12,13,14,13,12,15,13,14,12,13,14],[11,12,13,14,13,12,15,13,14,12,11,14],[11,12,14,14,12,12,15,13,14,12,13,14],[11,12,13,12,13,12,15,15,14,12,12,14]]
#}
lc_feature_data = {}
with open('yth2.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = key.strip()
            value = eval(value.strip())
            lc_feature_data[key] = value

#第三个人
#person3_feature_data = {
#    '1850': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
#    '1851': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
#    '1852': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]]
#}
zlf_feature_data = {}
with open('lrj.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = key.strip()
            value = eval(value.strip())
            zlf_feature_data[key] = value

#第四个人
#person4_feature_data = {
#    '1850': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
#    '1851': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
#    '1852': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]]
#}
lrj_feature_data = {}
with open('zlf.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = key.strip()
            value = eval(value.strip())
            lrj_feature_data[key] = value

#第5个人
#person4_feature_data = {
#    '1850': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
#    '1851': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]],
#    '1852': [[1,2,33,42,35,28,54,32,46,25,3,4],[14,22,35,43,34,22,55,32,4,26,12,46],[12,23,45,41,28,22,51,38,4,32,38,64],[15,2,33,28,39,26,75,52,43,21,25,34]]
#}
lhy_feature_data = {}
with open('lc.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = key.strip()
            value = eval(value.strip())
            lhy_feature_data[key] = value


# 获取增强的样本数量
num_augmentations = 100

#数据增强
zlf_augmented_data = augment_data(zlf_feature_data, num_augmentations=num_augmentations)
lc_augmented_data = augment_data(lc_feature_data, num_augmentations=num_augmentations)
yth_augmented_data = augment_data(zlf_feature_data, num_augmentations=num_augmentations)
lhy_augmented_data = augment_data(lc_feature_data, num_augmentations=num_augmentations)
lrj_augmented_data = augment_data(zlf_feature_data, num_augmentations=num_augmentations)

#print("增强后的 yth_feature_data 规模：", sum(len(samples) for samples in yth_augmented_data.values()))

# 构建 DataFrame
dfs = []
for person_data in [yth_feature_data, lc_feature_data, zlf_feature_data,lrj_feature_data,lhy_feature_data]:
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

# 创建 imputer，使用均值填充 NaN
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 对特征数据进行填充 NaN 处理
X_imputed = imputer.fit_transform(X)

for _ in range(num_runs):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=random.randint(1, 100))

    # SVM 模型
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_y_pred = svm_model.predict(X_test)
    svm_report = classification_report(y_test, svm_y_pred, output_dict=True)
    svm_macro_avg.append(svm_report['macro avg']['f1-score'])
    svm_weighted_avg.append(svm_report['weighted avg']['f1-score'])
    svm_accuracies.append(accuracy_score(y_test, svm_y_pred))

    # 随机森林模型
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_report = classification_report(y_test, rf_y_pred, output_dict=True)
    rf_macro_avg.append(rf_report['macro avg']['f1-score'])
    rf_weighted_avg.append(rf_report['weighted avg']['f1-score'])
    rf_accuracies.append(accuracy_score(y_test, rf_y_pred))

    # 贝叶斯模型
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_y_pred = nb_model.predict(X_test)
    nb_report = classification_report(y_test, nb_y_pred, output_dict=True)
    nb_macro_avg.append(nb_report['macro avg']['f1-score'])
    nb_weighted_avg.append(nb_report['weighted avg']['f1-score'])
    nb_accuracies.append(accuracy_score(y_test, nb_y_pred))

# Macro Avg 图
plt.figure(figsize=(8, 6))  # 创建新的图形窗口
plt.plot(range(1, num_runs+1), svm_macro_avg, marker='o', label='SVM Macro Avg')
plt.plot(range(1, num_runs+1), rf_macro_avg, marker='s', label='Random Forest Macro Avg')
plt.plot(range(1, num_runs+1), nb_macro_avg, marker='x', label='Naive Bayes Macro Avg')
plt.title('Macro Avg Scores')
plt.xlabel('Run')
plt.ylabel('Macro Avg Score')
plt.legend()
plt.grid(True)
plt.xticks(range(1, num_runs+1))

plt.tight_layout()
plt.show()

# Weighted Avg 图
plt.figure(figsize=(8, 6))  # 创建新的图形窗口
plt.plot(range(1, num_runs+1), svm_weighted_avg, marker='o', label='SVM Weighted Avg')
plt.plot(range(1, num_runs+1), rf_weighted_avg, marker='s', label='Random Forest Weighted Avg')
plt.plot(range(1, num_runs+1), nb_weighted_avg, marker='x', label='Naive Bayes Weighted Avg')
plt.title('Weighted Avg Scores')
plt.xlabel('Run')
plt.ylabel('Weighted Avg Score')
plt.legend()
plt.grid(True)
plt.xticks(range(1, num_runs+1))

plt.tight_layout()
plt.show()

# Accuracy 图
plt.figure(figsize=(8, 6))  # 创建新的图形窗口
plt.plot(range(1, num_runs+1), svm_accuracies, marker='o', label='SVM Accuracy')
plt.plot(range(1, num_runs+1), rf_accuracies, marker='s', label='Random Forest Accuracy')
plt.plot(range(1, num_runs+1), nb_accuracies, marker='x', label='Naive Bayes Accuracy')
plt.title('Accuracy Scores')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(range(1, num_runs+1))

plt.tight_layout()
plt.show()

