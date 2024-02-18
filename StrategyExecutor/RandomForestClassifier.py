# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2024-01-30 15:24
:last_date:
    2024-01-30 15:24
:description:
    
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

origin_data_path = '../daily_all_2024/1.txt'
# origin_data_path = '../daily_all_100_bad_0.5/1.txt'
# origin_data_path = '../daily_all_100_bad_0.3/1.txt'
# origin_data_path = '../daily_all_100_bad_0.0/1.txt'
# data = load_data('../daily_data_exclude_new_can_buy_with_back/龙洲股份_002682.txt')
data = pd.read_csv(origin_data_path, low_memory=False)
# data = pd.read_csv('../daily_all_100_bad_0.3/1.txt', low_memory=False)#675229
# data = pd.read_csv('../daily_all_2024/1.txt', low_memory=False)
signal_columns = [column for column in data.columns if 'signal' in column]
# 获取data中在signal_columns中的列
X = data[signal_columns]
# 获取data中'Days Held'列，如果值大于1就为False，否则为True
y = data['Days Held'] <= 1
true_count = sum(y)
print(f'true_count: {true_count/len(y)}')
is_skip = True

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
    'n_estimators': 100,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_depth': 10
}

# 根据参数创建模型文件的名称
model_filename = f'rf_classifier_{best_params["n_estimators"]}_{best_params["min_samples_split"]}_{best_params["min_samples_leaf"]}_{"None" if best_params["max_depth"] is None else best_params["max_depth"]}.joblib'
model_filename = os.path.join('../model/', model_filename)
# 检查预训练模型是否存在
if is_skip and os.path.exists(model_filename):
    # 如果存在，直接加载模型
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

# 假设阈值设置为0.7，仅考虑预测概率大于此值的树
threshold = 0.7

# 获取每棵树的预测结果
tree_preds = np.array([tree.predict_proba(X_test) for tree in rf_classifier.estimators_])

# 初始化高置信度投票计数器
high_confidence_votes = np.zeros((X_test.shape[0], 2))

# 对每棵树的预测进行迭代，仅当预测概率超过阈值时才计入投票
for tree_pred in tree_preds:
    # 对于每个样本，找到概率最高的类别及其概率
    max_probs = np.max(tree_pred, axis=1)
    max_class = np.argmax(tree_pred, axis=1)

    # 仅考虑概率大于阈值的预测
    for i, (prob, cls) in enumerate(zip(max_probs, max_class)):
        if prob > threshold:
            high_confidence_votes[i, cls] += 1

# 根据高置信度投票决定最终类别
y_pred_high_conf = np.argmax(high_confidence_votes, axis=1)

# 计算准确率
accuracy_high_conf = accuracy_score(y_test, y_pred_high_conf)
print(f'高置信度模型准确率: {accuracy_high_conf}')

# 获取更详细的分类报告（如精确度、召回率、F1分数）
report_high_conf = classification_report(y_test, y_pred_high_conf)
print(report_high_conf)