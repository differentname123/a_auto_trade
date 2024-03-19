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
import random
import time
from math import ceil
from multiprocessing import Process, Pool

import numpy as np
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import train_test_split
from joblib import dump, load
import cupy as cp
import cudf
from sklearn.model_selection import ParameterGrid

from StrategyExecutor.common import downcast_dtypes

D_MODEL_PATH = '/mnt/d/model/all_models/'
G_MODEL_PATH = '/mnt/g/model/all_models/'
MODEL_PATH = '/mnt/w/project/python_project/a_auto_trade/model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
MODEL_OTHER = '../model/other'
MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/reports'

def train_and_dump_model(clf, X_train, y_train, model_file_path, exist_model_file_path):
    """
    训练模型并保存到指定路径
    :param clf: 分类器实例
    :param X_train: 训练数据集特征
    :param y_train: 训练数据集标签
    :param model_path: 模型保存路径
    :param model_name: 模型名称
    """
    start_time = time.time()
    model_name = os.path.basename(model_file_path)
    out_put_path = model_file_path
    if not os.path.exists(os.path.dirname(out_put_path)):
        os.makedirs(os.path.dirname(out_put_path))
    print(f"开始训练模型: {model_name}")
    clf.fit(X_train, y_train)
    dump(clf, model_file_path)
    print(f"耗时 {time.time() - start_time} 模型已保存: {model_file_path}\n\n")
    with open(exist_model_file_path, 'a') as f:
        f.write(model_name + '\n')

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
    true_ratio = y_train.mean()
    param_grid = {
        'RandomForest': {
            'n_estimators': [100, 250, 300, 400, 500, 600],
            'max_depth': [10, 20, 30, 40, 100, 200, 400],
            'min_samples_split': [2, 3, 4, 5, 6],
            'min_samples_leaf': [1, 2, 3, 4]
        }
    }[model_type]

    params_list = list(ParameterGrid(param_grid))
    random.shuffle(params_list)
    print(f"待训练的模型数量: {len(params_list)}")
    save_path = G_MODEL_PATH
    for params in params_list:
        model_name = f"{model_type}_origin_data_path_dir_{origin_data_path_dir}_thread_day_{thread_day}_true_ratio_{true_ratio}_{'_'.join([f'{key}_{value}' for key, value in params.items()])}.joblib"
        model_file_path = os.path.join(save_path, origin_data_path_dir, model_name)
        flag = False
        for model_path in MODEL_PATH_LIST:
            exist_model_file_path = os.path.join(model_path, 'existed_model.txt')
            if is_skip and os.path.exists(exist_model_file_path):
                with open(exist_model_file_path, 'r') as f:
                     # 读取每一行存入existed_model_list，去除换行符
                    existed_model_list = [line.strip() for line in f]
                    if model_name in existed_model_list:
                        print(f"模型已存在，跳过: {model_name}")
                        flag = True
                        break
        if flag:
            continue
        exist_model_file_path = os.path.join(save_path, 'existed_model.txt')
        clf = RandomForestClassifier(**params)
        train_and_dump_model(clf, X_train, y_train, model_file_path, exist_model_file_path)

def train_all_model(file_path_path, profit=1, thread_day_list=None, is_skip=True):
    """
    为file_path_path生成各种模型
    :param file_path_path: 数据集路径
    :param thread_day_list: 判断为True的天数列表
    :param is_skip: 如果已有模型，是否跳过
    """
    if thread_day_list is None:
        thread_day_list = [1, 2, 3]
    origin_data_path_dir = os.path.dirname(file_path_path)
    origin_data_path_dir = origin_data_path_dir.split('/')[-1]
    print("加载数据{}...".format(file_path_path))
    data = cudf.read_csv(file_path_path)
    memory = data.memory_usage(deep=True).sum()
    print(f"原始数据集内存: {memory / 1024 ** 2:.2f} MB")
    signal_columns = [column for column in data.columns if '信号' in column]
    X = data[signal_columns]
    X = downcast_dtypes(X)
    memory = X.memory_usage(deep=True).sum()
    print(f"转换后数据集内存: {memory / 1024 ** 2:.2f} MB")
    ratio_result_path = os.path.join(MODEL_OTHER, origin_data_path_dir + 'ratio_result.json')
    try:
        with open(ratio_result_path, 'r') as f:
            ratio_result = json.load(f)
    except FileNotFoundError:
        ratio_result = {}
    for thread_day in thread_day_list:
        key_name = f'后续{thread_day}日最高价利润率'
        y = data[key_name] >= profit
        ratio_key = origin_data_path_dir + '_' + str(thread_day)
        if ratio_key in ratio_result:
            true_ratio = ratio_result[ratio_key]
        else:
            true_ratio = y.mean()
            ratio_result[ratio_key] = true_ratio
            with open(ratio_result_path, 'w') as f:
                json.dump(ratio_result, f)
        print(f"处理天数阈值: {thread_day}, 真实比率: {true_ratio:.4f}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_models(X_train, y_train, 'RandomForest', thread_day, true_ratio, is_skip, origin_data_path_dir)


def build_models():
    """
    训练所有模型
    """
    origin_data_path_list = [
        '../train_data/profit_1_day_1_bad_0.2/bad_0.2_data_batch_count.csv',
        '../train_data/profit_1_day_2_bad_0.2/bad_0.2_data_batch_count.csv',
        '../train_data/profit_1_day_1_bad_0.3/bad_0.3_data_batch_count.csv',
        '../train_data/profit_1_day_2_bad_0.3/bad_0.3_data_batch_count.csv',
        '../train_data/profit_1_day_1_bad_0.4/bad_0.4_data_batch_count.csv',
        '../train_data/profit_1_day_2_bad_0.4/bad_0.4_data_batch_count.csv',
        '../train_data/profit_1_day_1_bad_0.5/bad_0.5_data_batch_count.csv',
        '../train_data/profit_1_day_2_bad_0.5/bad_0.5_data_batch_count.csv']
    for origin_data_path in origin_data_path_list:
        train_all_model(origin_data_path, profit=1, thread_day_list=[1,2], is_skip=True)

if __name__ == '__main__':
    build_models()