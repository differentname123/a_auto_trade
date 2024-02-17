# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2024-01-30 15:24
:last_date:
    2024-01-30 15:24
:description:
    
"""
import json

from imblearn.over_sampling import SMOTE

import os

from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 指定后端为 'Agg'

import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict

from StrategyExecutor.common import load_data
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

# origin_data_path = '../daily_all_2024/1.txt'
#
origin_data_path = '../daily_all_100_bad_0.0/1.txt'
# origin_data_path = '../daily_all_100_bad_0.3/1.txt'
# origin_data_path = '../daily_all_100_bad_0.5/1.txt'
# 获取origin_data_path的上一级目录，不要更上一级目录
origin_data_path_dir = os.path.dirname(origin_data_path)
origin_data_path_dir = origin_data_path_dir.split('/')[-1]
# data = load_data('../daily_data_exclude_new_can_buy_with_back/龙洲股份_002682.txt')
# data = pd.read_csv('../daily_all_100_bad_0.5/1.txt', low_memory=False)
# data = pd.read_csv('../daily_all_100_bad_0.3/1.txt', low_memory=False)
data = pd.read_csv(origin_data_path, low_memory=False)
signal_columns = [column for column in data.columns if 'signal' in column]
# 获取data中在signal_columns中的列
X = data[signal_columns]
# 获取data中'Days Held'列，如果值大于1就为False，否则为True
y = data['Days Held'] <= 1
# 计算y中True的数量
true_count = sum(y)
print(f'true_count: {true_count/len(y)}')

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = X_train.astype(int)
X_test = X_test.astype(int)
# 处理类别不平衡
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# 使用随机森林分类器作为基模型
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
step_ratio = 0.01
# RFECV 实例，使用 3 折交叉验证
selector = RFECV(model, step=step_ratio, cv=3, n_jobs=-1, scoring='accuracy')
selector.fit(X_train, y_train)

# 获取被选择和被剔除的特征
selected_features = X_train.columns[selector.support_]
removed_features = X_train.columns[~selector.support_]

# 输出被选择和被剔除的特征
print("Selected features:", selected_features)
print("Removed features:", removed_features)

# 保存被选择和被剔除的特征到文件
with open('../model/selected_features.txt', 'w') as file:
    file.write("Selected features:\n")
    file.write('\n'.join(selected_features))

with open('../model/removed_features.txt', 'w') as file:
    file.write("Removed features:\n")
    file.write('\n'.join(removed_features))

# 输出剔除特征后的模型性能
print("Optimal number of features:", selector.n_features_)
print("Best cross-validation score:", max(selector.cv_results_['mean_test_score']))

# 绘制交叉验证分数图
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score")

# 计算每一步的特征数量
initial_num_features = X_train.shape[1]
 # 与 RFECV 中的 step 参数相同
number_of_features = [initial_num_features - int(round(step_ratio * i * initial_num_features))
                      for i in range(len(selector.cv_results_['mean_test_score']))]
print(number_of_features)
print(selector.cv_results_['mean_test_score'])
plt.plot(number_of_features, selector.cv_results_['mean_test_score'])
plt.savefig('../model/cross_validation_score_plot.png')  # 保存图像