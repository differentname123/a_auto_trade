# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2024-01-30 15:24
:last_date:
    2024-01-30 15:24
:description:
    通用的一些关于随机森林模型的代码
"""
import os

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# origin_data_path = '../daily_all_2024/1.txt'
# origin_data_path = '../train_data/26_daily_all_1_bad_0.3/1.txt'
# origin_data_path = '../train_data/daily_all_1_bad_0.0/1.txt'
origin_data_path = '../temp/real_time_price.csv'
# origin_data_path = '../train_data/daily_all_1_bad_0.0/1_all.txt'
# origin_data_path = '../daily_all_2024/1.txt'
# origin_data_path = '../daily_all_100_bad_0.3/1.txt'
# origin_data_path = '../daily_all_100_bad_0.0/1.txt'
# data = load_data('../daily_data_exclude_new_can_buy_with_back/龙洲股份_002682.txt')
data = pd.read_csv(origin_data_path, low_memory=False, dtype={'代码': str})
signal_columns = [column for column in data.columns if 'signal' in column]
# 筛选出data中日期大于2024-01-26并且小于2024-02-05的数据
# data = data[(data['日期'] > '2024-02-20')]
data = data[(data['日期'] <= '2024-02-20')]


# data = pd.read_csv('../daily_all_100_bad_0.3/1.txt', low_memory=False)#675229
# data = pd.read_csv('../daily_all_2024/1.txt', low_memory=False)
# signal_columns = [column for column in data.columns if 'X' in column]

# 获取data中在signal_columns中的列
X = data[signal_columns]
# 获取data中去除signal_columns中的列
X1 = data.drop(signal_columns, axis=1)
# 获取data中'Days Held'列，如果值大于1就为False，否则为True
data['Days Held'] = 1
y = data['Days Held'] <= 1
# true_count = sum(y)
# print(f'true_count: {true_count/len(y)}')
is_skip = True
# # 找到X中每列值都为True的行
# rows_with_all_true = X.all(axis=True)
# result = X[rows_with_all_true]
# # 将y合并到result中，按照index合并，行数以result为准
# result = pd.concat([result, y], axis=1)


model_filename = '../model/all_models/smote_RandomForest_origin_data_path_dir_daily_all_100_bad_0.5_thread_day_1_true_ratio_0.3322392519068618_max_depth_None_min_samples_leaf_1_min_samples_split_4_n_estimators_250.joblib'
rf_classifier = load(model_filename)
X_test = X
y_test = y

# 使用模型预测概率
y_pred_proba = rf_classifier.predict_proba(X_test)

# 测试集的总样本数
total_samples = len(y_test)

# 设置阈值的范围，比如从0.5增加到0.9，步长为0.05
threshold_values = np.arange(0.85, 1, 0.01)


for threshold in threshold_values:
    # 获取高于阈值的预测结果
    high_confidence_true = (y_pred_proba[:, 1] > threshold)

    # 计算这些高置信度预测的精确率
    selected_true = high_confidence_true & y_test  # 这里修正了逻辑，确保我们正确地选取了预测为True且实际为True的样本
    precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(high_confidence_true) > 0 else 0

    # 高于阈值的预测样本数
    predicted_true_samples = np.sum(high_confidence_true)

    print(f'阈值: {threshold:.2f}, 高置信度预测为True的精确率: {precision:.2f}, 预测为True的样本数量: {predicted_true_samples}, 总样本数: {total_samples}')

    # 如果有高于阈值的预测样本，打印对应的原始数据
    if predicted_true_samples > 0:
        # 直接使用布尔索引从原始data中选择对应行
        selected_samples = X1[high_confidence_true]
        # 统计selected_samples中 收盘 的和
        close_sum = selected_samples['收盘'].sum()
        print(f'高于阈值 {threshold:.2f} 的预测样本对应的原始数据:{close_sum}')
        print(selected_samples.head())  # 为了简洁起见，这里只打印前几行数据
        # 将 代码 和 收盘 用,连接写入文件../final_zuhe/select/select_temp_real_time_order.txt
        # selected_samples[['代码', '收盘']].to_csv('../final_zuhe/select/select_temp_real_time_order.txt', index=False, mode='a', header=False, sep=',')
        # 统计selected_samples中每个日期 对应的数量
        print(selected_samples['日期'].value_counts())