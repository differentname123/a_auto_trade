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
import json
import os

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
MODEL_PATH = '../model/all_models'

def get_thread_data(data, rf_classifier, threshold):
    """
    使用随机森林模型预测data中的数据，如果预测结果高于阈值threshold，则打印对应的原始数据，返回高于阈值的原始数据
    :param data:
    :param rf_classifier:
    :param threshold:
    :return:
    """
    selected_samples = None
    signal_columns = [column for column in data.columns if 'signal' in column]
    # 获取data中在signal_columns中的列
    X = data[signal_columns]
    # 获取data中去除signal_columns中的列
    X1 = data.drop(signal_columns, axis=1)
    y_pred_proba = rf_classifier.predict_proba(X)
    high_confidence_true = (y_pred_proba[:, 1] > threshold)
    predicted_true_samples = np.sum(high_confidence_true)
    # 如果有高于阈值的预测样本，打印对应的原始数据
    if predicted_true_samples > 0:
        # 直接使用布尔索引从原始data中选择对应行
        selected_samples = X1[high_confidence_true]
        # 统计selected_samples中 收盘 的和
        close_sum = selected_samples['收盘'].sum()
        print(f'高于阈值 {threshold:.2f} 的预测样本对应的原始数据:{close_sum} 代码:{set(selected_samples["代码"].values)}')
        print(selected_samples['日期'].value_counts())
    return selected_samples

def load_rf_model(model_path):
    """
    加载随机森林模型
    :param model_path:
    :return:
    """
    all_rf_model_list = []
    output_filename = '../temp/all_model_reports.json'
    # 加载output_filename，找到最好的模型
    with open(output_filename, 'r') as file:
        sorted_scores = json.load(file)
        for model_name, score, threshold in sorted_scores:
            if score > 5 and 'thread_day_2' in model_name:
                all_rf_model_map = {}
                try:
                    model = load(os.path.join(model_path, model_name))
                    all_rf_model_map[model_name] = model
                    all_rf_model_map['threshold'] = threshold
                    all_rf_model_map['score'] = score
                    all_rf_model_list.append(all_rf_model_map)
                except FileNotFoundError:
                    print(f"模型 {model_name} 不存在，跳过。")
                    return
    return all_rf_model_list

def get_all_good_data(data):
    """
    获取所有模型的预测结果，如果预测结果高于阈值，则打印对应的原始数据
    :param data:
    :param all_rf_model_list:
    :return:
    """
    all_rf_model_list = load_rf_model(MODEL_PATH)
    all_selected_samples = []
    for rf_model_map in all_rf_model_list:
        rf_model = rf_model_map[list(rf_model_map.keys())[0]]
        threshold = rf_model_map['threshold']
        print(f"模型 {rf_model} 的阈值为 {threshold:.2f}")
        selected_samples = get_thread_data(data, rf_model, threshold)
        if selected_samples is not None:
            all_selected_samples.append(selected_samples)
    # 如果all_selected_samples不为空，将所有的selected_samples合并
    if len(all_selected_samples) > 0:
        all_selected_samples = pd.concat(all_selected_samples)
    return all_selected_samples

def get_all_good_data_with_model_list(data, all_rf_model_list):
    """
    获取所有模型的预测结果，如果预测结果高于阈值，则打印对应的原始数据
    :param data:
    :param all_rf_model_list:
    :return:
    """
    all_selected_samples = []
    for rf_model_map in all_rf_model_list:
        rf_model = rf_model_map[list(rf_model_map.keys())[0]]
        threshold = rf_model_map['threshold']
        score = rf_model_map['score']
        model_name = list(rf_model_map.keys())[0]
        if threshold > 1:
            threshold = 0.98
        print(f"模型 {rf_model} 的阈值为 {threshold:.2f}")
        selected_samples = get_thread_data(data, rf_model, threshold)
        if selected_samples is not None:
            selected_samples['score'] = score
            selected_samples['model_name'] = model_name
            all_selected_samples.append(selected_samples)
    # 如果all_selected_samples不为空，将所有的selected_samples合并
    if len(all_selected_samples) > 0:
        all_selected_samples = pd.concat(all_selected_samples)
    return all_selected_samples

if __name__ == '__main__':
    origin_data_path = '../temp/real_time_price.csv'
    data = pd.read_csv(origin_data_path, low_memory=False, dtype={'代码': str})
    get_all_good_data(data)

