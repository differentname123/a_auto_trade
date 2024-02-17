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
import os
from multiprocessing import Process, Pool

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

MODEL_PATH = '../model/all_models'


def train_and_dump_model(clf, X_train, y_train, model_path, model_name):
    """
    训练模型并保存到指定路径
    :param clf: 分类器实例
    :param X_train: 训练数据集特征
    :param y_train: 训练数据集标签
    :param model_path: 模型保存路径
    :param model_name: 模型名称
    """
    print(f"开始训练模型: {model_name}")
    clf.fit(X_train, y_train)
    dump(clf, os.path.join(model_path, model_name))
    print(f"模型已保存: {os.path.join(model_path, model_name)}")
    # get_model_report(model_path, model_name)

def train_all_model(file_path_path, thread_day_list=None, is_skip=True):
    """
    为file_path_path生成各种模型
    :param file_path_path: 数据集路径
    :param thread_day_list: 判断为True的天数列表
    :param is_skip: 如果已有模型，是否跳过
    """
    if thread_day_list is None:
        thread_day_list = [1, 2, 3]
    # 获取origin_data_path的上一级目录，不要更上一级目录
    origin_data_path_dir = os.path.dirname(file_path_path)
    origin_data_path_dir = origin_data_path_dir.split('/')[-1]
    print("加载数据{}...".format(file_path_path))
    data = pd.read_csv(file_path_path, low_memory=False)
    signal_columns = [column for column in data.columns if 'signal' in column]
    X = data[signal_columns]

    for thread_day in thread_day_list:
        y = data['Days Held'] <= thread_day
        true_ratio = sum(y) / len(y)
        print(f"处理天数阈值: {thread_day}, 真实比率: {true_ratio:.4f}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_models(X_train, y_train, 'RandomForest', thread_day, true_ratio, is_skip, origin_data_path_dir)
        train_models(X_train, y_train, 'GradientBoosting', thread_day, true_ratio, is_skip, origin_data_path_dir)

def train_models(X_train, y_train, model_type, thread_day, true_ratio, is_skip, origin_data_path_dir):
    """
    训练指定类型的模型并处理类别不平衡
    :param X_train: 训练数据集特征
    :param y_train: 训练数据集标签
    :param model_type: 模型类型 ('RandomForest' 或 'GradientBoosting')
    :param thread_day: 天数阈值
    :param true_ratio: 真实比率
    :param is_skip: 是否跳过已存在的模型
    :param data: 完整数据集
    :param signal_columns: 信号列
    """
    param_grid = {
        'RandomForest': {
            'n_estimators': [100, 250, 300, 400, 500, 600],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 3, 4, 5, 6],
            'min_samples_leaf': [1, 2, 3, 4]
        },
        'GradientBoosting': {
            'n_estimators': [100, 250, 300, 400, 500, 600],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }[model_type]

    for params in ParameterGrid(param_grid):
        model_name = f"{model_type}_origin_data_path_dir_{origin_data_path_dir}_thread_day_{thread_day}_true_ratio_{true_ratio}_{'_'.join([f'{key}_{value}' for key, value in params.items()])}.joblib"
        if is_skip and os.path.exists(os.path.join(MODEL_PATH, model_name)):
            print(f"模型 {model_name} 已存在，跳过训练。")
            continue
        clf = RandomForestClassifier(**params) if model_type == 'RandomForest' else GradientBoostingClassifier(**params)
        train_and_dump_model(clf, X_train, y_train, MODEL_PATH, model_name)

        # 对于类别不平衡的处理
        print("处理类别不平衡...")
        # 检查并转换布尔列为整数
        X_train = X_train.astype(int)
        # 现在使用SMOTE应该不会报错
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        model_name_smote = f"smote_{model_name}"
        train_and_dump_model(clf, X_train_smote, y_train_smote, MODEL_PATH, model_name_smote)

def get_model_report(model_path, model_name):
    """
    为单个模型生成报告，并更新模型报告文件
    :param model_path: 模型路径
    :param model_name: 当前正在处理的模型名称
    """
    file_path_list = ['../daily_all_2024/1.txt', '../daily_all_100_bad_0.5/1.txt', '../daily_all_100_bad_0.3/1.txt']

    report_path = os.path.join(model_path, 'a_model_report.json')
    result_dict = {}
    # 加载已有报告
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            result_dict = json.load(f)
    print(f"为模型生成报告 {model_name}...")
    thread_ratio = 0.95
    new_temp_dict = {}


    model = load(os.path.join(model_path, model_name))
    # 提取模型的天数阈值
    thread_day = int(model_name.split('thread_day_')[1].split('_')[0])
    threshold_values = np.arange(0.5, 1, 0.05)

    for file_path in file_path_list:
        # 判断result_dict[model_name][file_path]是否存在，如果存在则跳过
        if model_name in result_dict and file_path in result_dict[model_name]:
            print(f"模型 {model_name} 对于文件 {file_path} 的报告已存在，跳过。")
            continue
        temp_dict_list = []
        data = pd.read_csv(file_path, low_memory=False)
        signal_columns = [column for column in data.columns if 'signal' in column]
        X_test = data[signal_columns]
        y_test = data['Days Held'] <= thread_day
        total_samples = len(y_test)
        y_pred_proba = model.predict_proba(X_test)
        for threshold in threshold_values:
            temp_dict = {}
            high_confidence_true = (y_pred_proba[:, 1] > threshold)
            selected_true = high_confidence_true & y_test
            precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(high_confidence_true) > 0 else 0
            predicted_true_samples = np.sum(high_confidence_true)

            temp_dict['threshold'] = float(threshold)  # 确保阈值也是原生类型
            temp_dict['precision'] = precision
            temp_dict['predicted_true_samples'] = int(predicted_true_samples)
            temp_dict['total_samples'] = int(total_samples)  # 确保转换为Python原生int类型
            temp_dict['predicted_ratio'] = predicted_true_samples / total_samples if total_samples > 0 else 0
            temp_dict['score'] = precision * temp_dict['predicted_ratio'] * 100 if precision > thread_ratio else 0

            temp_dict_list.append(temp_dict)

        # 按照score排序
        temp_dict_list = sorted(temp_dict_list, key=lambda x: x['score'], reverse=True)
        new_temp_dict[file_path] = temp_dict_list

    result_dict[model_name] = new_temp_dict
    # 将结果保存到文件
    with open(report_path, 'w') as f:
        json.dump(result_dict, f)
    print(f"模型报告已生成: {model_name}\n\n")

    return result_dict

def build_models():
    """
    训练所有模型
    """
    origin_data_path_list = ['../daily_all_2024/1.txt', '../daily_all_100_bad_0.5/1.txt', '../daily_all_100_bad_0.3/1.txt', '../daily_all_100_bad_0.0/1.txt']
    for origin_data_path in origin_data_path_list:
        train_all_model(origin_data_path, [1, 2], is_skip=True)


def get_all_model_report():
    """
    使用多进程获取所有模型的报告。
    """
    while True:
        model_path = MODEL_PATH
        model_list = [model for model in os.listdir(model_path) if model.endswith('.joblib')]

        # 使用进程池来并行处理每个模型的报告生成
        with Pool(1) as p:
            p.starmap(get_model_report, [(model_path, model_name) for model_name in model_list])


# 将build_models和get_all_model_report用两个进程同时执行
if __name__ == '__main__':
    p1 = Process(target=build_models)
    p2 = Process(target=get_all_model_report)
    p1.start()
    p2.start()
    p1.join()
    p2.join()