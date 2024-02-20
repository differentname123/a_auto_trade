# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2024-01-30 15:24
:last_date:
    2024-01-30 15:24
:description:
    针对策略26的训练
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
# origin_data_path = '../daily_all_2024/1.txt'
# origin_data_path = '../daily_all_100_bad_0.3/1.txt'
origin_data_path = '../daily_all_100_bad_0.0/1.txt'
# data = load_data('../daily_data_exclude_new_can_buy_with_back/龙洲股份_002682.txt')
data = pd.read_csv(origin_data_path, low_memory=False)
# data = pd.read_csv('../daily_all_100_bad_0.3/1.txt', low_memory=False)#675229
# data = pd.read_csv('../daily_all_2024/1.txt', low_memory=False)
# signal_columns = [column for column in data.columns if 'X' in column]
signal_columns = [column for column in data.columns if 'signal' in column]
# 获取data中在signal_columns中的列
X = data[signal_columns]
# 获取data中'Days Held'列，如果值大于1就为False，否则为True
y = data['Days Held'] <= 1
true_count = sum(y)
print(f'true_count: {true_count/len(y)}')
is_skip = True
# # 找到X中每列值都为True的行
# rows_with_all_true = X.all(axis=True)
# result = X[rows_with_all_true]
# # 将y合并到result中，按照index合并，行数以result为准
# result = pd.concat([result, y], axis=1)


# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 处理类别不平衡
X_train = X_train.astype(int)
X_test = X_test.astype(int)
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train = X_train_smote
y_train = y_train_smote


# 您找到的最佳参数
# best_params = {
#     'n_estimators': 300,
#     'min_samples_split': 5,
#     'min_samples_leaf': 2,
#     'max_depth': None
# }

# best_params = {
#     'n_estimators': 300,
#     'min_samples_split': 2,
#     'min_samples_leaf': 1,
#     'max_depth': None
# }

best_params = {
    'n_estimators': 400,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_depth': 20
}

# 根据参数创建模型文件的名称
model_filename = f'rf_classifier_{best_params["n_estimators"]}_{best_params["min_samples_split"]}_{best_params["min_samples_leaf"]}_{"None" if best_params["max_depth"] is None else best_params["max_depth"]}.joblib'
model_filename = os.path.join('../model/', model_filename)
# 检查预训练模型是否存在
if is_skip and os.path.exists(model_filename):
    # 如果存在，直接加载模型
    model_filename = '../model/all_models/RandomForest_origin_data_path_dir_daily_all_100_bad_0.5_thread_day_1_true_ratio_0.3322392519068618_max_depth_None_min_samples_leaf_3_min_samples_split_4_n_estimators_100.joblib'
    rf_classifier = load(model_filename)
    X_test = X
    y_test = y
else:
    # 如果不存在，创建并训练模型
    rf_classifier = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_depth=best_params['max_depth'],
        random_state=42
    )

    # # 执行交叉验证
    # cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=3)
    #
    # # 输出每次交叉验证的结果和平均分数
    # print("CV Scores: ", cv_scores)
    # print("Average CVScore: ", cv_scores.mean())

    # 使用您的训练数据训练模型
    rf_classifier.fit(X_train, y_train)

    # 保存模型
    dump(rf_classifier, model_filename)

# 使用模型预测概率
y_pred_proba = rf_classifier.predict_proba(X_test)

# 使用模型进行类别预测而不是概率预测
y_pred = rf_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy}')

# 获取更详细的分类报告（如精确度、召回率、F1分数）
report = classification_report(y_test, y_pred)
print(report)

# 测试集的总样本数
total_samples = len(y_test)

# 设置阈值的范围，比如从0.5增加到0.9，步长为0.05
threshold_values = np.arange(0.5, 1, 0.05)

for threshold in threshold_values:
    # 获取高于阈值的预测结果
    high_confidence_true = (y_pred_proba[:, 1] > threshold)

    # 计算这些高置信度预测的精确率
    selected_true = high_confidence_true[y_test]  # 选取实际为 True 的样本
    precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(high_confidence_true) > 0 else 0

    # 高于阈值的预测样本数
    predicted_true_samples = np.sum(high_confidence_true)

    print(f'阈值: {threshold:.2f}, 高置信度预测为True的精确率: {precision:.2f}, 预测为True的样本数量: {predicted_true_samples}, 总样本数: {total_samples}')

    # # 如果有高于阈值的预测样本，打印对应的原始数据
    # if predicted_true_samples > 0:
    #     # 从测试集中选择高于阈值的样本
    #     selected_samples = X_test[high_confidence_true]
    #     # 获取selected_samples的索引，找到data中对应的原始数据
    #     selected_samples = data.iloc[selected_samples.index]
    #     print(f'高于阈值 {threshold:.2f} 的预测样本对应的原始数据:')
    #     print(selected_samples)