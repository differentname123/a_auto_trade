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

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# X_train = X_train.astype(int)
# X_test = X_test.astype(int)
# # 处理类别不平衡
# smote = SMOTE()
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 创建梯度提升机分类器
gbm_classifier = GradientBoostingClassifier(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 250, 300, 400, 500, 600],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'learning_rate': [0.01, 0.1, 0.2]
}

# 结果文件路径
results_file = '{}{}GradientBoostingClassifier_reports.json'.format('../model/reports/', origin_data_path_dir)

# 读取现有结果（如果文件存在）
if os.path.exists(results_file):
    with open(results_file, 'r') as infile:
        existing_results = json.load(infile)
        processed_params = {result['params_key'] for result in existing_results}
else:
    existing_results = []
    processed_params = set()

# 自定义阈值
threshold_range = np.arange(0.5, 1.05, 0.05)
count = 0

# 对参数网格进行迭代
for params in ParameterGrid(param_grid):
    params_key = '_'.join([f'{key}_{value}' for key, value in params.items()])
    count += 1
    # 检查参数是否已处理
    if params_key in processed_params:
        print(f'Already processed {params_key}')
        continue  # 如果已处理，则跳过

    # 设置梯度提升机分类器参数
    gbm_classifier.set_params(**params)

    # 进行模型训练和预测
    proba_predictions = cross_val_predict(gbm_classifier, X_train, y_train, cv=3, n_jobs=-1, method='predict_proba')

    one_report = {'params': params, 'params_key': params_key, 'report': []}
    for threshold in threshold_range:
        # 应用自定义阈值来确定分类
        custom_predictions = (proba_predictions[:, 1] >= threshold).astype(int)
        # 获取分类报告
        report = classification_report(y_train, custom_predictions, output_dict=True, zero_division=0)
        report['threshold'] = threshold
        one_report['report'].append(report)
    # 根据 'True' 类别的 'recall' 和 'precision' 乘积值排序，仅当 'precision' > 0.8
    one_report['report'].sort(key=lambda x: x.get('True', {}).get('recall', 0) * x.get('True', {}).get('precision', 0) if x.get('True',{}).get('precision', 0) > 0.9 else 0, reverse=True)

    # 将新结果追加到现有结果中
    existing_results.append(one_report)

    print(f'Processed {count} of {one_report}')
    # 根据每个 one_report 的最大 recall 和 precision 乘积值排序 existing_results
    existing_results.sort(key=lambda x: max((r.get('True', {}).get('recall', 0) * r.get('True', {}).get('precision',0) if r.get('True', {}).get('precision', 0) > 0.9 else 0) for r in x['report']), reverse=True)

    # 保存结果到JSON文件
    with open(results_file, 'w') as outfile:
        json.dump(existing_results, outfile)