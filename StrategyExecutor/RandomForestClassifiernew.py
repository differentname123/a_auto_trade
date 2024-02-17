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
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict

from StrategyExecutor.common import load_data
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
# origin_data_path = '../daily_all_2024/1.txt'
# origin_data_path = '../daily_all_100_bad_0.3/1.txt'
origin_data_path = '../daily_all_100_bad_0.5/1.txt'
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
# 定义自定义评分函数
def true_accuracy_score(y_true, y_pred):
    """
    计算针对类别 'True' 的准确率。
    假设 'True' 类别标记为 1。
    """
    true_positive = ((y_pred == 1) & (y_true == 1)).sum()
    true_total = (y_true == 1).sum()
    return true_positive / true_total if true_total else 0

# 创建自定义评分对象
true_accuracy_scorer = make_scorer(true_accuracy_score, greater_is_better=True)

# 数据预处理
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 处理类别不平衡
X_train, y_train = SMOTE().fit_resample(X_train.astype(int), y_train)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 250, 300, 400, 500, 600],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3, 4]
}

# 设置并执行网格搜索
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring=true_accuracy_scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 保存网格搜索对象
grid_search_filename = '{}{}classification_reports.joblib'.format('../model/search/', origin_data_path_dir)
dump(grid_search, grid_search_filename)
print(f"GridSearchCV object saved to {grid_search_filename}")

# 提取并打印最佳参数
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# 获取并保存最佳模型
best_rf_classifier = grid_search.best_estimator_
model_filename = '{}{}best_random_forest_model.joblib'.format('../model/', origin_data_path_dir)
dump(best_rf_classifier, model_filename)
print(f"Model saved to {model_filename}")

# 可选：在测试集上评估模型
y_pred = best_rf_classifier.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))