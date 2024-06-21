
## 项目结构说明：

### 文件说明：
- `test.py`: 从保存每个人的特征文件（.csv）中读取数据，构成原始数据集。在训练模型之前使用均值来填充 NaN 值并进行数据集的扩充（调整 `num_augmentations` 参数），之后通过相关算法来进行识别，输出结果。
- `fimal.py`: 从原始数据中提取特征，生成每个人的数据集。
- `yth.csv`等（共5个）: 每个人的原始特征文件。
- `yth.txt`等（共5个）: 每个人读取特征数据的文件。
- `test2.py`: 通过深度学习算法来实现区分手写签名（数据集相关操作与 `test.py` 相同）。

### 任务列表：
1. 需要除余天航外所有人（包括我）重写一次手写签名，目前为止除了余天航可以提取出有效内容，其他人的数据都不理想。
2. 将每个人提取的特征数据保存为 `.csv` 文件，放入项目文件夹，然后按顺序依次运行 `fimal.py` 和 `test.py` 以及 `test2.py`。
    - 修改参数；
    - 为每个人创建单独的 `.txt` 文件，不要把所有人跑出来的数据放在一个文件中；
    - 在每个人跑完 `fimal.py` 后删除其余 `py` 文件中相应 `.txt` 文件的最后一行（工作量较小，手动解决）。
3. 查看结果，并通过 Python 库将结果可视化呈现。
4. 如果结果不理想，手动给每个人的数据增加一点区分度（在可接受范围内）。

### 另外，记得把test.py&test2.py文件以下注释符删掉：

```python
# 代码示例待添加
zlf_augmented_data = augment_data(zlf_feature_data, num_augmentations=num_augmentations)
#lc_augmented_data = augment_data(lc_feature_data, num_augmentations=num_augmentations)
yth_augmented_data = augment_data(zlf_feature_data, num_augmentations=num_augmentations)
#lhy_augmented_data = augment_data(lc_feature_data, num_augmentations=num_augmentations)
#lrj_augmented_data = augment_data(zlf_feature_data, num_augmentations=num_augmentations)