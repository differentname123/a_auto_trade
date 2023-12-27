# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2023-12-25 4:25
:last_date:
    2023-12-25 4:25
:description:
    决策树
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 生成数据集
X = np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1]])
y = np.array([1, 1, 0, 0])

# 构建决策树
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# 查找叶子节点
leaves = clf.apply(X)

# 获取叶子节点的详细信息
for leaf_idx in np.unique(leaves):
    leaf_data = X[leaves == leaf_idx]
    leaf_info = clf.tree_.children_left[leaf_idx], clf.tree_.children_right[leaf_idx], clf.tree_.threshold[leaf_idx], clf.tree_.value[leaf_idx]
    print(f"叶子节点{leaf_idx}的详细信息：{leaf_info}")

