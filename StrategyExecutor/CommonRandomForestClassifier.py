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
from collections import Counter

import os
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第二张GPU，索引从0开始
import random
import sys
import psutil
import gc  # 引入垃圾回收模块
from functools import partial

import concurrent.futures

import json
import multiprocessing

import time
import traceback
import warnings
from multiprocessing import Pool
from multiprocessing import Process, current_process
from itertools import chain, zip_longest
import shutil
import math
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from concurrent.futures import ThreadPoolExecutor, as_completed

from StrategyExecutor.common import low_memory_load

D_MODEL_PATH = '/mnt/d/model/all_models/'
G_MODEL_PATH = '/mnt/g/model/all_models/'
MODEL_PATH = '/mnt/w/project/python_project/a_auto_trade/model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
warnings.filterwarnings(action='ignore', category=UserWarning)


def get_thread_data_new_tree_1(tree_preds, X1, tree_threshold=0, cha_zhi_threshold=0, abs_threshold=0):
    """
    使用随机森林模型预测data中的数据，如果预测结果高于阈值threshold，则打印对应的原始数据，返回高于阈值的原始数据
    :param data:
    :param rf_classifier:
    :param threshold:
    :return:
    """
    selected_samples = None
    n_trees = tree_preds.shape[0]
    true_counts = np.sum(tree_preds[:, :, 1] > tree_threshold, axis=0)
    # 计算预测概率大于阈值的次数，判断为False
    false_counts = np.sum(tree_preds[:, :, 0] > tree_threshold, axis=0)
    if true_counts.size == 0 and false_counts.size == 0:
        print('没有预测结果大于阈值')
        return selected_samples

    # 计算大于阈值判断为True的概率
    true_proba = true_counts / n_trees
    # 计算大于阈值判断为False的概率
    false_proba = false_counts / n_trees
    proba_df = np.column_stack((false_proba, true_proba))
    y_pred_proba = proba_df
    if abs_threshold > 0:
        threshold = abs_threshold
        high_confidence_true = (y_pred_proba[:, 1] > threshold)
        predicted_true_samples = np.sum(high_confidence_true)
        # 如果有高于阈值的预测样本，打印对应的原始数据
        if predicted_true_samples > 0:
            # 直接使用布尔索引从原始data中选择对应行
            selected_samples = X1[high_confidence_true].copy()
            # 统计selected_samples中 收盘 的和
            close_sum = selected_samples['收盘'].sum()
            selected_samples.loc[:, 'thread'] = y_pred_proba[high_confidence_true, 1]
            selected_samples.loc[:, 'basic_thread'] = threshold
            print(f'高于阈值 {threshold:.2f} 的预测样本对应的原始数据:{close_sum} 代码:{set(selected_samples["代码"].values)} 收盘最小值:{selected_samples["收盘"].min()} 收盘最大值:{selected_samples["收盘"].max()}')
            # print(selected_samples['日期'].value_counts())
        return selected_samples
    else:
        if cha_zhi_threshold > 0:
            proba_diff = y_pred_proba[:, 1] - y_pred_proba[:, 0]

            # 判断概率差异是否大于阈值
            high_confidence_true = (proba_diff > cha_zhi_threshold)
            predicted_true_samples = np.sum(high_confidence_true)
            # 如果有高于阈值的预测样本，打印对应的原始数据
            if predicted_true_samples > 0:
                # 直接使用布尔索引从原始data中选择对应行
                selected_samples = X1[high_confidence_true].copy()
                # 统计selected_samples中 收盘 的和
                close_sum = selected_samples['收盘'].sum()
                selected_samples.loc[:, 'thread'] = y_pred_proba[high_confidence_true, 1]
                selected_samples.loc[:, 'basic_thread'] = cha_zhi_threshold
                print(f'高于阈值 {cha_zhi_threshold:.2f} 的预测样本对应的原始数据:{close_sum} 代码:{set(selected_samples["代码"].values)} 收盘最小值:{selected_samples["收盘"].min()} 收盘最大值:{selected_samples["收盘"].max()}')
                # print(selected_samples['日期'].value_counts())
            return selected_samples


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
        selected_samples = X1[high_confidence_true].copy()
        # 统计selected_samples中 收盘 的和
        close_sum = selected_samples['收盘'].sum()
        selected_samples.loc[:, 'thread'] = y_pred_proba[high_confidence_true, 1]
        selected_samples.loc[:, 'basic_thread'] = threshold
        print(f'高于阈值 {threshold:.2f} 的预测样本对应的原始数据:{close_sum} 代码:{set(selected_samples["代码"].values)} 收盘最小值:{selected_samples["收盘"].min()} 收盘最大值:{selected_samples["收盘"].max()}')
        # print(selected_samples['日期'].value_counts())
    return selected_samples


def load_rf_model(need_load=True):
    """
    加载随机森林模型
    :param model_path:
    :return:
    """
    all_rf_model_list = []
    output_filename = '../final_zuhe/other/all_model_reports.json'
    # 加载output_filename，找到最好的模型
    with open(output_filename, 'r') as file:
        sorted_scores = json.load(file)
        for model_name, score, threshold in sorted_scores:
            for model_path in MODEL_PATH_LIST:
                model_file_path = os.path.join(model_path, model_name)
                if os.path.exists(model_file_path):
                    break
            if score > 5 and 'thread_day_2' in model_name:
                all_rf_model_map = {}
                try:
                    if need_load:
                        model = load(model_file_path)
                        all_rf_model_map[model_name] = model
                    else:
                        all_rf_model_map[model_name] = model_file_path
                        all_rf_model_map['model_path'] = model_file_path
                    all_rf_model_map['threshold'] = threshold
                    all_rf_model_map['score'] = score
                    all_rf_model_list.append(all_rf_model_map)
                except FileNotFoundError:
                    print(f"模型 {model_name} 不存在，跳过。")
    print(f"加载了 {len(all_rf_model_list)} 个模型")
    return all_rf_model_list


def split_model_info_list(all_model_info_list, split_count):
    # 先对列表按照model_size降序排序
    sorted_model_info_list = sorted(all_model_info_list, key=lambda x: x['model_size'], reverse=True)

    # 初始化split_count个子列表
    split_lists = [[] for _ in range(split_count)]
    # 初始化每个子列表的model_size之和
    size_sums = [0] * split_count

    # 遍历所有模型信息
    for model_info in sorted_model_info_list:
        # 找出当前model_size之和最小的子列表
        min_size_index = size_sums.index(min(size_sums))
        # 将当前模型信息添加到该子列表中
        split_lists[min_size_index].append(model_info)
        # 更新该子列表的model_size之和
        size_sums[min_size_index] += model_info['model_size']

    # 打印每组的大小和model_size之和
    for i, split_list in enumerate(split_lists):
        group_size = len(split_list)
        group_size_sum = size_sums[i]
        print(f"Group {i + 1}: {group_size} models, total model_size: {group_size_sum:.2f}")

    return split_lists


def balance_disk(class_key='/mnt/w'):
    """
    平衡两个磁盘中的数据大小
    :param class_key: 用于识别模型路径的关键字
    :return: 更新后的模型信息列表
    """
    final_output_filename = '../final_zuhe/other/all_data.json'

    # 读取模型信息
    with open(final_output_filename, 'r') as file:
        all_model_info_list = json.load(file)

    # 将模型信息列表按照model_size排序
    all_model_info_list_sorted = sorted(all_model_info_list, key=lambda x: x['model_size'], reverse=True)

    # 初始化两个组的总大小
    total_size_w = 0
    total_size_other = 0

    # 分配模型到两个组，尽量保持总size相等
    w_models = []
    other_models = []
    for model_info in all_model_info_list_sorted:
        if total_size_w <= total_size_other:
            w_models.append(model_info)
            total_size_w += model_info['model_size']
        else:
            other_models.append(model_info)
            total_size_other += model_info['model_size']

    # 移动文件的函数
    def move_file(model_info, target_path):
        src_path = model_info['model_path']
        dst_dir = os.path.join(target_path, os.path.dirname(src_path.split('all_models')[1]).replace('/', ''))
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.move(src_path, dst_path)
        model_info['model_path'] = dst_path
        print(f"Moved {src_path} to {dst_path}")

    # 创建线程池进行文件移动操作
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 移动w_models中的文件
        for model_info in w_models:
            if class_key not in model_info['model_path']:
                executor.submit(move_file, model_info, MODEL_PATH)

        # 移动other_models中的文件
        for model_info in other_models:
            if class_key in model_info['model_path']:
                executor.submit(move_file, model_info, D_MODEL_PATH)

    # 合并两个列表
    all_model_info_list = w_models + other_models
    # 输出平衡后的每组模型大小和模型个数
    print(
        f"Total size of w_models: {total_size_w:.2f} GB, {len(w_models)} models in w_models Total size of other_models: {total_size_other:.2f} GB, {len(other_models)} models in other_models")

    # 将更新后的列表写回文件
    with open(final_output_filename, 'w') as file:
        json.dump(all_model_info_list, file)

    return all_model_info_list

def get_all_train_data(model_name_list):
    """
    获取所有的训练数据
    :param model_name_list:
    :return:
    """
    all_train_data = []
    for model_name in model_name_list:
        all_train_data.append(model_name.split('origin_data_path_dir_')[1].split('_bad_0_train')[0])
    return list(set(all_train_data))

def load_rf_model_new(date_count_threshold=100, need_filter=True, need_balance=False, model_max_size=10000000,
                      model_min_size=0, abs_threshold=1):
    """
    加载随机森林模型
    :param model_path:
    :return:
    """


    output_filename = '../final_zuhe/other/all_model_reports_cuml.json'

    not_estimated_model_list = []
    not_estimated_model_list_back = f'../final_zuhe/other/not_estimated_model_list.txt'
    model_file_list = []
    for model_path in MODEL_PATH_LIST:
        # 获取model_path下的所有文件的全路径，如果是目录还需要继续递归
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if 'good' in root:
                    continue
                model_file_list.append(os.path.join(root, file))
    # 加载output_filename，找到最好的模型
    with open(output_filename, 'r') as file:
        sorted_scores_list = json.load(file)
        # 获取所有的模型名称
        model_name_list = [sorted_scores['model_name'] for sorted_scores in sorted_scores_list]
        train_data_list = get_all_train_data(model_name_list)
        train_data_list.append('all')
        thread_day_list = ['train_thread_day_2', 'train_thread_day_1', 'all']
        # 将sorted_scores_list转换成dataframe然后按照model_name分组
        df = pd.DataFrame(sorted_scores_list)
        temp_df = df.copy()
        for train_data in train_data_list:
            for thread_day in thread_day_list:
                final_output_filename = f'../final_zuhe/other/new_good_all_model_reports_cuml_{train_data}_{thread_day}.json'
                exist_stocks = set()
                all_rf_model_list = []
                if thread_day == 'all':
                    current_df = temp_df
                else:
                    # 过滤出model_name包含thread_day的数据
                    current_df = temp_df[temp_df['model_name'].str.contains(thread_day)]
                if train_data != 'all':
                    # 过滤出model_name包含train_data的数据
                    current_df = current_df[current_df['model_name'].str.contains(train_data)]
                else:
                    current_df = current_df
                sorted_scores_list = current_df.groupby('model_name')

                for model_name, sorted_scores in sorted_scores_list:
                    # 如果sorted_scores存在date_count为0的情况，直接跳过
                    if sorted_scores['date_count'].min() == 0:
                        print(f"模型 {model_name} 的date_count为0，跳过。")
                        continue
                    # 获取abs_threshold最大的那行数据
                    sorted_scores = sorted_scores.sort_values(by='abs_threshold', ascending=False)
                    sorted_scores = sorted_scores.iloc[0]
                    if sorted_scores['date_count'] <= date_count_threshold:
                        print(f"模型 {sorted_scores['model_name']} 的date_count小于{date_count_threshold}，跳过。")
                        continue
                    if sorted_scores['abs_threshold'] > abs_threshold:
                        print(f"模型 {model_name} 的阈值大于{abs_threshold}，跳过。")
                        continue
                    model_name = sorted_scores['model_name']
                    model_size = sorted_scores['model_size']
                    model_file_path = None
                    for model_path in model_file_list:
                        if model_name in model_path:
                            model_file_path = model_path
                            break

                    if need_filter:
                        current_stocks = set(sorted_scores['true_stocks_set'])
                        if len(current_stocks - exist_stocks) == 0:
                            print(f"模型 {model_name} 已经有相似的模型，跳过。")
                            continue
                        exist_stocks = exist_stocks | current_stocks
                    if model_file_path is not None:
                        # 判断model_file_path大小是否大于model_max_size
                        if model_size > model_max_size or model_size < model_min_size:
                            print(f"{os.path.getsize(model_file_path)}大小超过 {model_max_size}G，跳过。")
                            continue
                        if sorted_scores['date_count'] > date_count_threshold:
                            sorted_scores['true_stocks_set'] = []
                            # 将sorted_scores转换成字典
                            sorted_scores = sorted_scores.to_dict()
                            other_dict = sorted_scores
                            other_dict['model_path'] = model_file_path
                            other_dict['model_size'] = model_size
                            all_rf_model_list.append(other_dict)
                    else:
                        not_estimated_model_list.append(model_name)
                        print(f"模型 {model_name} 不存在，跳过。")
                print(f"加载了 {len(all_rf_model_list)} 个模型")
                # 将all_rf_model_list按照model_size从小到大排序
                all_rf_model_list.sort(key=lambda x: x['date_count'], reverse=True)
                # 将all_rf_model_list存入final_output_filename
                with open(final_output_filename, 'w') as file:
                    json.dump(all_rf_model_list, file)
                with open(not_estimated_model_list_back, 'w') as file:
                    for model_name in not_estimated_model_list:
                        file.write(model_name + '\n')
                if need_balance:
                    all_rf_model_list = balance_disk()
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


def get_all_good_data_with_model_list(data, all_rf_model_list, plus_threshold=0.05):
    """
    获取所有模型的预测结果，如果预测结果高于阈值，则打印对应的原始数据
    :param data:
    :param all_rf_model_list:
    :return:
    """
    print(f"加载了 {len(all_rf_model_list)} 个模型 plus_threshold={plus_threshold}")
    all_selected_samples = []
    count = 0
    for rf_model_map in all_rf_model_list:
        start = time.time()
        rf_model = rf_model_map[list(rf_model_map.keys())[0]]
        threshold = rf_model_map['threshold'] + plus_threshold
        score = rf_model_map['score']
        model_name = list(rf_model_map.keys())[0]
        if threshold > 1:
            threshold = 0.98
        # print(f"模型 {rf_model} 的阈值为 {threshold:.2f}")
        selected_samples = get_thread_data(data, rf_model, threshold)
        if selected_samples is not None:
            selected_samples['score'] = score
            selected_samples['model_name'] = model_name
            all_selected_samples.append(selected_samples)
        count += 1
        print(
            f"整体进度 {count}/{len(all_rf_model_list)} score {score} threshold {threshold}  basic_threshold {rf_model_map['threshold']} 耗时 {time.time() - start} 模型 {model_name}\n\n")
    # 如果all_selected_samples不为空，将所有的selected_samples合并
    if len(all_selected_samples) > 0:
        all_selected_samples = pd.concat(all_selected_samples)
    return all_selected_samples


def get_all_good_data_with_model_name_list_old(data, plus_threshold=0.05):
    """
    获取所有模型的预测结果，如果预测结果高于阈值，则打印对应的原始数据
    :param data:
    :param all_rf_model_list:
    :return:
    """
    all_rf_model_list = load_rf_model(need_load=False)
    print(f"加载了 {len(all_rf_model_list)} 个模型 plus_threshold={plus_threshold}")
    all_selected_samples = []
    count = 0
    for rf_model_map in all_rf_model_list:
        try:
            start = time.time()
            threshold = rf_model_map['threshold'] + plus_threshold
            score = rf_model_map['score']
            model_name = list(rf_model_map.keys())[0]
            rf_model = load(rf_model_map['model_path'])
            if threshold > 1:
                threshold = 0.98
            # print(f"模型 {rf_model} 的阈值为 {threshold:.2f}")
            selected_samples = get_thread_data(data, rf_model, threshold)
            if selected_samples is not None:
                selected_samples['score'] = score
                selected_samples['model_name'] = model_name
                all_selected_samples.append(selected_samples)
        except Exception as e:
            print(f"模型 {model_name} 加载失败 {e}")
        finally:
            count += 1
            print(
                f"整体进度 {count}/{len(all_rf_model_list)} score {score} threshold {threshold}  basic_threshold {rf_model_map['threshold']} 耗时 {time.time() - start} 模型 {model_name}\n\n")
    # 如果all_selected_samples不为空，将所有的selected_samples合并
    if len(all_selected_samples) > 0:
        all_selected_samples = pd.concat(all_selected_samples)
    return all_selected_samples


def process_model(rf_model_map, data, plus_threshold=0.05):
    """
    处理单个模型的预测并收集数据
    :param rf_model_map: 单个随机森林模型的信息
    :param data: 需要预测的数据
    :param plus_threshold: 阈值调整
    :return: 选中的样本（如果有）
    """
    try:
        start = time.time()
        threshold = rf_model_map['threshold'] + plus_threshold
        score = rf_model_map['score']
        model_name = list(rf_model_map.keys())[0]
        rf_model = load(rf_model_map['model_path'])
        for key, value in rf_model_map.items():
            if 'tree' in key:
                tree_0_list = []
                if 'tree_0' in key:
                    if value['cha_zhi_threshold'] != 0:
                        tree_0_list.append(value)

        selected_samples = get_thread_data(data, rf_model, threshold)
        if selected_samples is not None:
            selected_samples['score'] = score
            selected_samples['model_name'] = model_name
            return selected_samples
    except Exception as e:
        print(f"模型 {model_name} 加载失败 {e}")
    finally:
        elapsed_time = time.time() - start
        print(f"模型 {model_name} 耗时 {elapsed_time}")


def get_proba_data_tree(data, rf_classifier):
    signal_columns = [column for column in data.columns if '信号' in column]
    # 获取data中在signal_columns中的列
    X = data[signal_columns]
    # 获取data中去除signal_columns中的列
    X1 = data.drop(signal_columns, axis=1)
    tree_preds = np.array([tree.predict_proba(X) for tree in rf_classifier.estimators_])
    return tree_preds, X1


def get_all_good_data_with_model_name_list(data, plus_threshold=0.05):
    all_rf_model_list = load_rf_model(need_load=False)
    print(f"加载了 {len(all_rf_model_list)} 个模型 plus_threshold={plus_threshold}")

    # 使用Pool对象来并行处理
    with Pool(processes=multiprocessing.cpu_count() - 15) as pool:  # 可以调整processes的数量以匹配你的CPU核心数量
        results = pool.starmap(process_model, [(model, data, plus_threshold) for model in all_rf_model_list])

    # 过滤掉None结果并合并DataFrame
    all_selected_samples = pd.concat([res for res in results if res is not None])

    return all_selected_samples


def get_thread_data_new_tree_0(y_pred_proba, X1, min_day=0, abs_threshold=0):
    """
    使用随机森林模型预测data中的数据，如果预测结果高于阈值threshold，则打印对应的原始数据，返回高于阈值的原始数据
    :param data:
    :param rf_classifier:
    :param threshold:
    :return:
    """
    selected_samples = None
    result_list = []
    debug = True
    if abs_threshold > 0:
        for cha in range(10, 11):
            cha = cha / 100
            threshold = abs_threshold - cha
            high_confidence_true = (y_pred_proba[1] >= threshold)
            predicted_true_samples = np.sum(high_confidence_true)
            # 如果有高于阈值的预测样本，打印对应的原始数据
            if predicted_true_samples > min_day:
                # 直接使用布尔索引从原始data中选择对应行
                selected_samples = X1[high_confidence_true].copy()
                if debug:
                    # 将对应的预测概率添加到selected_samples中
                    selected_samples.loc[:, 'cha_thread'] = y_pred_proba[high_confidence_true][1] - abs_threshold
                # 统计selected_samples中 收盘 的和
                result_list.append(selected_samples)
        if len(result_list) > 0:
            selected_samples = pd.concat(result_list)
            close_sum = selected_samples['收盘'].sum()
            # print(f'高于阈值 {threshold:.2f} 的预测样本对应的原始数据:{close_sum} 代码:{set(selected_samples["代码"].values)} 收盘最小值:{selected_samples["收盘"].min()} 收盘最大值:{selected_samples["收盘"].max()}')
            # print(selected_samples['日期'].value_counts())
    return selected_samples


def get_proba_data(data, rf_classifier):
    signal_columns = [column for column in data.columns if '信号' in column]
    drop_columns = [column for column in data.columns if '信号' in column or '股价' in column or 'MACD' in column]
    # 获取data中在signal_columns中的列
    X = data[signal_columns]
    # 获取data中去除signal_columns中的列
    X1 = data.drop(drop_columns, axis=1)
    y_pred_proba = rf_classifier.predict_proba(X)
    return y_pred_proba, X1


def process_model_new(rf_model_map, data):
    """
    处理单个模型的预测并收集数据
    :param rf_model_map: 单个随机森林模型的信息
    :param data: 需要预测的数据
    :param plus_threshold: 阈值调整
    :return: 选中的样本（如果有）
    """
    start = time.time()
    all_selected_samples_list = []
    model_name = os.path.basename(rf_model_map['model_path'])
    load_time = 0
    selected_samples_size = 0
    try:
        # 如果rf_model_map['model_path']是文件路径，则加载模型
        if 'model' not in rf_model_map:
            rf_model = load(rf_model_map['model_path'])
        else:
            rf_model = rf_model_map['model']
            print(f"模型 {model_name} 已经加载")
        load_time = time.time() - start

        value = rf_model_map
        y_pred_proba, X1 = get_proba_data(data, rf_model)

        selected_samples = get_thread_data_new_tree_0(y_pred_proba, X1, min_day=value['min_day'],
                                                      abs_threshold=value['abs_threshold'])
        if selected_samples is not None:
            selected_samples['date_count'] = value['date_count']
            selected_samples['model_name'] = model_name
            all_selected_samples_list.append(selected_samples)
            selected_samples_size = selected_samples.shape[0]

        if len(all_selected_samples_list) > 0:
            return pd.concat(all_selected_samples_list)
        return None
    except Exception as e:
        traceback.print_exc()
        print(f"模型 {model_name} 加载失败 {e}")
    finally:
        elapsed_time = time.time() - start
        print(f"模型 {model_name} 耗时 {elapsed_time} 加载耗时 {load_time} 选中样本数 {selected_samples_size}")


def model_worker(model_info_list, data, result_list, max_workers=1):
    print(f"Process {current_process().name} started.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_model_new, model_info, data): model_info for model_info in model_info_list}
        for future in as_completed(futures):
            model_info = futures[future]
            try:
                result = future.result()
                if result is not None:
                    result_list.append(result)
            except Exception as exc:
                error_message = f"Model {model_info['model_path']} generated an exception: {exc}"
                print(error_message)
                traceback.print_exc()
                # 可以选择将错误信息添加到结果列表中，或者将其记录到文件或日志系统中
    print(f"Process {current_process().name} finished.")


def compress_model_thread(model_path, compress=3):
    """
    压缩单个模型文件，并返回压缩前后的大小信息。

    :param model_path: 模型文件的绝对路径
    :param compress: 压缩级别（0-9），其中0是无压缩，9是最大压缩
    :return: 压缩前后的文件大小和路径
    """
    try:
        original_size = os.path.getsize(model_path)
        model = load(model_path)

        # 定义临时压缩模型文件路径
        compressed_model_path = model_path + '.compressed'

        # 使用joblib的dump方法来压缩模型到临时文件
        dump(model, compressed_model_path, compress=compress)

        # 获取压缩后的文件大小
        compressed_size = os.path.getsize(compressed_model_path)

        # 删除原始模型文件
        os.remove(model_path)
        # 重命名临时压缩文件，替换原始文件
        os.rename(compressed_model_path, model_path)

        return model_path, original_size, compressed_size
    except Exception as e:
        print(f"模型 {model_path} 压缩失败 {e}")
        return model_path, 0, 0


def compress_model(compress=1):
    """
    压缩模型文件
    """
    model_paths = []
    for model_path in MODEL_PATH_LIST:
        # 获取所有模型的文件名
        for root, ds, fs in os.walk(model_path):
            for f in fs:
                if f.endswith('.joblib'):
                    full_name = os.path.join(root, f)
                    # 获取文件的大小
                    size = os.path.getsize(full_name)
                    if size > 3 * 1024 ** 3:
                        model_paths.append(full_name)

    print(f"共有 {len(model_paths)} 个模型文件")
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(compress_model_thread, model_path, compress) for model_path in model_paths]
        for future in futures:
            model_path, original_size, compressed_size = future.result()
            print(f"Compressed {model_path}:")
            print(f" - Original size: {original_size / 1024 / 1024:.2f} MB")
            print(f" - Compressed size: {compressed_size / 1024 / 1024:.2f} MB")


def interleave_lists(list1, list2):
    """
    交错合并两个列表。如果列表长度不相同，较长列表的额外元素将出现在交错合并的列表末尾。

    :param list1: 第一个列表
    :param list2: 第二个列表
    :return: 交错合并后的列表
    """
    # zip_longest会在较短的列表结束后用None填充，chain.from_iterable将结果扁平化
    return [item for item in chain.from_iterable(zip_longest(list1, list2)) if item is not None]


def pre_load_model(max_memory=50000):
    """
    提前加载一部分模型到内存中，以便于加速选股
    :return:
    """
    start = time.time()
    base_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    now_memory = base_memory
    with open('../final_zuhe/other/good_all_model_reports_cuml_0.1.json', 'r') as file:
        all_model_info_list = json.load(file)
    # 将all_model_info_list按照model_size从小到大排序
    all_model_info_list.sort(key=lambda x: x['model_size'], reverse=False)
    count = 0
    all_model_info_list = all_model_info_list[:10]
    for model_info in all_model_info_list:
        if (now_memory - base_memory) > max_memory:
            break
        model = load(model_info['model_path'])
        model_info['model'] = model
        # 获取model的内存占用,单位为MB，如果内存占用超过max_memory，则不再加载
        # 获取模型的内存占用，单位为MB
        count += 1
        now_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"加载了 {count}个 模型，内存占用 {now_memory:.2f}MB")
    print(f"加载了 {count} 个模型，总耗时 {time.time() - start}")
    # time.sleep(50)
    return all_model_info_list


def get_all_good_data_with_model_name_list_new_pre(data, all_model_info_list, date_count_threshold=50, code_list=[]):
    # 获取data的最大和最小日期，保留到天,并且拼接为字符串
    date_str = f"{data['日期'].min().strftime('%Y%m%d')}_{data['日期'].max().strftime('%Y%m%d')}"

    # all_model_info_list = pre_load_model()
    # with open('../final_zuhe/other/good_all_model_reports_cuml.json', 'r') as file:
    #     all_model_info_list = json.load(file)
    start = time.time()
    # 将code_list加入到每个模型中
    for model_info in all_model_info_list:
        model_info['code_list'] = code_list
    # 将all_model_info_list按照model_path分类，包含‘/mnt/w’的为一类，其余为一类
    all_model_info_list_w = [model_info for model_info in all_model_info_list if '/mnt/w' in model_info['model_path']]
    all_model_info_list_other = [model_info for model_info in all_model_info_list if
                                 '/mnt/w' not in model_info['model_path']]
    all_model_info_list_w.sort(key=lambda x: x['model_size'], reverse=True)
    all_model_info_list_other.sort(key=lambda x: x['model_size'], reverse=False)
    # 将all_model_info_list_w和all_model_info_list_other交叉合并
    all_model_info_list = interleave_lists(all_model_info_list_w, all_model_info_list_other)
    print(
        f"大小平衡后 all_model_info_list_w {len(all_model_info_list_w)} all_model_info_list_other {len(all_model_info_list_other)}")

    print(f"总共加载了 {len(all_model_info_list)} 个模型，date_count_threshold={date_count_threshold}")
    # 存储最终结果的列表
    result_list = []
    code_result_list = []

    model_worker(all_model_info_list, data, result_list, code_result_list, 10)

    all_selected_samples = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    all_selected_samples.to_csv(f'../temp/data/all_selected_samples_{date_str}.csv', index=False)
    if code_list != []:
        code_result_list_samples = pd.concat(code_result_list,
                                             ignore_index=True) if code_result_list else pd.DataFrame()
        code_result_list_samples.to_csv(f'../temp/data/code_result_list_samples_{date_str}.csv', index=False)

    print(f"总耗时 {time.time() - start}")
    return all_selected_samples


def get_all_good_data_with_model_name_list_new(data, all_model_info_list, date_count_threshold=50,
                                               process_count=1, thread_count=1, output_file_path=None):
    """
    使用多进程加上多线程的方式获取all_model_info_list中模型的预测结果，最后将满足条件的数据合并保持
    :param data:
    :param all_model_info_list:
    :param date_count_threshold:
    :param process_count:
    :param thread_count:
    :return:
    """
    print(f'当前时间 {datetime.now()}')
    # 获取data的最大和最小日期，保留到天,并且拼接为字符串
    date_str = f"{data['日期'].min().strftime('%Y%m%d')}_{data['日期'].max().strftime('%Y%m%d')}"
    start = time.time()
    if output_file_path is None:
        output_file_path = f'../temp/data/second_all_selected_samples_{date_str}.csv'
    # 将all_model_info_list按照model_path分类，包含‘/mnt/w’的为一类，其余为一类
    all_model_info_list_w = [model_info for model_info in all_model_info_list if '/mnt/w' in model_info['model_path']]
    all_model_info_list_other = [model_info for model_info in all_model_info_list if
                                 '/mnt/w' not in model_info['model_path']]
    print(
        f"大小平衡后 all_model_info_list_w {len(all_model_info_list_w)} all_model_info_list_other {len(all_model_info_list_other)}")

    print(f"总共加载了 {len(all_model_info_list)} 个模型，date_count_threshold={date_count_threshold}")
    # 分割模型列表以分配给每个进程
    model_chunks_w = split_model_info_list(all_model_info_list_w, process_count)
    model_chunks_other = split_model_info_list(all_model_info_list_other, process_count)
    # 将model_chunks_w和model_chunks_other合并
    model_chunks = model_chunks_w + model_chunks_other

    # 创建进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as executor:
        # 存储最终结果的列表
        manager = multiprocessing.Manager()
        result_list = manager.list()

        # 提交任务到进程池
        futures = []
        jishu = 0
        for model_chunk in model_chunks:
            if jishu == 0:
                model_chunk.sort(key=lambda x: x['model_size'], reverse=True)
                jishu = 1
            else:
                model_chunk.sort(key=lambda x: x['model_size'], reverse=False)
                jishu = 0
            future = executor.submit(model_worker, model_chunk, data, result_list, thread_count)
            futures.append(future)

        # 等待所有任务完成
        concurrent.futures.wait(futures)

    all_selected_samples = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    all_selected_samples.to_csv(output_file_path, index=False)
    # mul_select(output_path)
    print(f"总耗时 {time.time() - start}")
    return all_selected_samples


def write_joblib_files_to_txt(directory):
    # 获取目录下所有以joblib结尾的文件,如果是目录还需要继续递归
    joblib_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.joblib'):
                joblib_files.append(file)

    # 将文件名写入existed_model.txt
    with open(os.path.join(directory, 'existed_model.txt'), 'w') as file:
        for joblib_file in joblib_files:
            file.write(joblib_file + '\n')

    print(f"以.joblib结尾的文件名已写入existed_model.txt文件。共写入{len(joblib_files)}个文件名。")


def delete_bad_model():
    with open('../final_zuhe/other/good_all_model_reports_cuml.json', 'r') as file:
        all_model_info_list = json.load(file)
        all_model_name_list = [model_info['model_name'] for model_info in all_model_info_list]
    with open('../final_zuhe/other/good_all_model_reports_cuml_0401压缩前的最好大于0的结果.json', 'r') as file:
        befor_all_model_info_list = json.load(file)
        befor_all_model_name_list = [model_info['model_name'] for model_info in befor_all_model_info_list]
    # 找出befor_all_model_name_list比all_model_name_list多出来的模型
    bad_model_list = list(set(befor_all_model_name_list) - set(all_model_name_list))
    print(f"共有 {len(bad_model_list)} 个模型需要删除")
    # 将bad_model_list写入bad_model_list.txt
    with open('../final_zuhe/other/bad_model_list.txt', 'w') as file:
        for bad_model in bad_model_list:
            file.write(bad_model + '\n')
    for model_path in MODEL_PATH_LIST:
        exist_model_path = os.path.join(model_path, 'existed_model.txt')
        with open(exist_model_path, 'r') as file:
            all_model_list = file.readlines()
            all_model_list = [model.strip() for model in all_model_list]
        # 删除all_model_list中的bad_model_list
        for bad_model in bad_model_list:
            if bad_model in all_model_list:
                all_model_list.remove(bad_model)
        # 将all_model_list写入existed_model.txt
        with open(exist_model_path, 'w') as file:
            for model in all_model_list:
                file.write(model + '\n')


def fix_param():
    data = pd.read_csv('../temp/code_result_list_samples_50.csv', low_memory=False, dtype={'代码': str})


def process_group(group):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    cha_zhi_all = 0
    cha_zhi_thread_day_1 = 0
    cha_zhi_others = 0
    temp_dict = {}
    total_row_thread_day_1 = 0
    total_row_others = 0
    true_counts_all = {f'true_count_{int(t * 100)}': 0 for t in thresholds}
    true_counts_thread_day_1 = {f'true_count_{int(t * 100)}': 0 for t in thresholds}
    true_counts_others = {f'true_count_{int(t * 100)}': 0 for t in thresholds}
    code_name = group['名称'].values[0]
    total_true_ratio_all = 0
    total_true_ratio_thread_day_1 = 0
    total_true_ratio_others = 0

    # 如果 '后续一日最高价利润率' 在group中，取第一个值
    profit_cols = ['后续1日最高价利润率', '后续2日最高价利润率', '后续3日最高价利润率']
    profit_values = [group[col].values[0] if col in group.columns else 0 for col in profit_cols]

    def process_row(row):
        param = eval(row['param'])
        abs_threshold = float(param['abs_threshold'])
        true_ratio = float(row['1'])
        model_name = param['model_name']

        # 将true_ratio保留两位小数，向下取整
        true_ratio = math.floor(true_ratio * 100) / 100
        precision_dict = param['precision_dict']
        # 尝试获取true_ratio对应的precision
        precision = precision_dict.get(str(true_ratio), 0)
        if true_ratio > 0.5 and precision == 0:
            precision = 1

        nonlocal total_true_ratio_all, total_true_ratio_thread_day_1, total_true_ratio_others, cha_zhi_all, cha_zhi_thread_day_1, cha_zhi_others
        nonlocal total_row_thread_day_1, total_row_others

        total_true_ratio_all += precision
        if 'thread_day_1' in model_name:
            total_true_ratio_thread_day_1 += precision
            total_row_thread_day_1 += 1
        else:
            total_true_ratio_others += precision
            total_row_others += 1

        for t in thresholds:
            if true_ratio > t:
                true_counts_all[f'true_count_{int(t * 100)}'] += 1
                if 'thread_day_1' in model_name:
                    true_counts_thread_day_1[f'true_count_{int(t * 100)}'] += 1
                else:
                    true_counts_others[f'true_count_{int(t * 100)}'] += 1

        cha_zhi = true_ratio - abs_threshold
        cha_zhi_all += cha_zhi
        if 'thread_day_1' in model_name:
            cha_zhi_thread_day_1 += cha_zhi
        else:
            cha_zhi_others += cha_zhi

    group.apply(process_row, axis=1)

    for t in thresholds:
        count_key = f'true_count_{int(t * 100)}'
        ratio_key = f'true_count_{int(t * 100)}_ratio'
        temp_dict[count_key] = true_counts_all[count_key]
        temp_dict[ratio_key] = true_counts_all[count_key] / group.shape[0] if group.shape[0] != 0 else 0
        temp_dict[f'{count_key}_thread_day_1'] = true_counts_thread_day_1[count_key]
        temp_dict[f'{ratio_key}_thread_day_1'] = true_counts_thread_day_1[
                                                     count_key] / total_row_thread_day_1 if total_row_thread_day_1 != 0 else 0
        temp_dict[f'{count_key}_others'] = true_counts_others[count_key]
        temp_dict[f'{ratio_key}_others'] = true_counts_others[
                                               count_key] / total_row_others if total_row_others != 0 else 0

    temp_dict['code'] = group['代码'].values[0]
    temp_dict['name'] = code_name
    temp_dict['cha_zhi_all'] = cha_zhi_all
    temp_dict['cha_zhi_thread_day_1'] = cha_zhi_thread_day_1
    temp_dict['cha_zhi_others'] = cha_zhi_others
    temp_dict['date'] = group['日期'].values[0]
    temp_dict['total_true_ratio_all'] = total_true_ratio_all
    temp_dict['total_true_ratio_thread_day_1'] = total_true_ratio_thread_day_1
    temp_dict['total_true_ratio_others'] = total_true_ratio_others
    for i, col in enumerate(profit_cols):
        temp_dict[col] = profit_values[i]

    return temp_dict


def analysis_model():
    # # 读取'../temp/cha_zhi_result.json'
    # with open('../temp/analysis_model/cha_zhi_result_2024-04-01.json', 'r', encoding='utf-8') as file:
    #     result_dict_list = json.load(file)
    # # 将result_dict_list按照true_count_ratio降序排序
    # result_dict_list = sorted(result_dict_list, key=lambda x: x['true_count_50_ratio'], reverse=True)
    # # 将result_dict_list转换为DataFrame
    # result_df = pd.DataFrame(result_dict_list)
    data = pd.read_csv('../temp/analysis_model/cha_zhi_result_2024-04-23.csv', low_memory=False, dtype={'code': str})

    # 遍历../temp/data目录下的所有文件，筛选出以code_result_list_samples_开头的文件
    data_files = []
    for root, dirs, files in os.walk('../temp/data'):
        for file in files:
            if file.startswith('code_result_list_samples_') and '20240423.csv' in file:
                full_name = os.path.join(root, file)
                data_files.append(full_name)
    for data_file in data_files:
        data = pd.read_csv(data_file, low_memory=False, dtype={'代码': str})
        # 获取所有的日期
        dates = data['日期'].unique()
        print(f"共有 {len(dates)} 个日期")

        for date in dates:
            # 筛选出当前日期的数据
            data_by_date = data[data['日期'] == date]

            # 将data_by_date按照 代码 分组，并对每个分组应用process_group函数
            result_dict_list = data_by_date.groupby(['代码']).apply(process_group).tolist()

            # 将result_dict_list按照cha_zhi_all降序排序
            result_dict_list = sorted(result_dict_list, key=lambda x: x['cha_zhi_all'], reverse=True)

            # 创建日期目录，如果不存在
            date_dir = f"../temp/analysis_model"
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            # 将result_dict_list转换为DataFrame
            result_df = pd.DataFrame(result_dict_list)
            # 将result_dict_list写入文件，注意不要中文乱码
            file_path = os.path.join(date_dir, f"cha_zhi_result_{date}.csv")
            result_df.to_csv(file_path, index=False, encoding='utf-8')
            print(f"日期 {date} 分析完成")


def test_load_memory_ratio(retries=3):
    """
    统计占用内存和加载时间的比例
    :param retries: 每个模型加载的次数
    :return:
    """
    result_dict_list = []
    with open('../final_zuhe/other/good_all_model_reports_cuml.json', 'r') as file:
        all_model_info_list = json.load(file)

    for model_info in all_model_info_list:
        temp_dict = {'model_name': model_info['model_name'], 'elapsed_time': [], 'memory_use': [], 'ratio': []}
        for _ in range(retries):
            gc.collect()  # 强制进行垃圾回收
            before_load_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            start = time.time()
            model = load(model_info['model_path'])
            elapsed_time = time.time() - start
            gc.collect()  # 再次进行垃圾回收，确保内存使用是由加载模型引起的
            memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_use = max(memory - before_load_memory, 0)  # 防止内存使用为负数
            ratio = elapsed_time / memory_use if memory_use else float('inf')

            temp_dict['elapsed_time'].append(elapsed_time)
            temp_dict['memory_use'].append(memory_use)
            temp_dict['ratio'].append(ratio)

        # Calculate the average of the retries
        avg_elapsed_time = sum(temp_dict['elapsed_time']) / retries
        avg_memory_use = sum(temp_dict['memory_use']) / retries
        avg_ratio = sum(temp_dict['ratio']) / retries

        print(
            f"模型 {model_info['model_name']} 平均加载耗时 {avg_elapsed_time:.2f}秒, 平均内存占用 {avg_memory_use:.2f}MB")

        # Save the averages
        temp_dict['elapsed_time'] = avg_elapsed_time
        temp_dict['memory_use'] = avg_memory_use
        temp_dict['ratio'] = avg_ratio

        result_dict_list.append(temp_dict)

    result_df = pd.DataFrame(result_dict_list)
    result_df.to_csv('../temp/load_memory_ratio.csv', index=False)


def analyze_data(df, key_list, target_key, top_n=10, abs_threshold_values=np.arange(0.5, 1, 0.01)):
    results = {}

    for key in key_list:
        # 对DataFrame中的key列进行降序排序
        sorted_df = df.sort_values(by=key, ascending=False)

        # 选出前[1,2,3,4,5,6]个数据，并计算target_key的相关统计数据
        for n in range(1, top_n + 1):
            top_n = sorted_df.head(n)
            count_greater_than_1 = (top_n[target_key] > 1).sum()
            count_less_or_equal_1 = (top_n[target_key] <= 1).sum()
            total_count = n

            key_name = f"{key}_top{n}"
            results[key_name] = {
                'success_ratio': round(count_greater_than_1 / total_count if total_count != 0 else 0, 2),
                'success_count': count_greater_than_1,
                'total_count': total_count,
                'fail_count': count_less_or_equal_1
            }

        # 选出值大于[0.8,0.9,0.95]的数据，并计算target_key的相关统计数据
        # 从0.8开始，每次增加0.01
        for threshold in abs_threshold_values:
            filtered_df = sorted_df[sorted_df[key] >= threshold]
            count_greater_than_1 = (filtered_df[target_key] > 1).sum()
            count_less_or_equal_1 = (filtered_df[target_key] <= 1).sum()
            total_count = filtered_df.shape[0]

            key_name = f"{key}_greater_than_{threshold}"
            results[key_name] = {
                'success_ratio': round(count_greater_than_1 / total_count if total_count != 0 else 0, 2),
                'success_count': count_greater_than_1,
                'total_count': total_count,
                'fail_count': count_less_or_equal_1
            }

    # 将results先按照total_count降序排序，再按照success_ratio降序排序
    results = {k: v for k, v in
               sorted(results.items(), key=lambda item: (item[1]['success_ratio'], item[1]['total_count']),
                      reverse=True)}
    return results


def get_thread():
    """
    获取相应的选股阈值 绝对值，相对值
    :return:
    """
    data_list = []
    for root, dirs, files in os.walk('../temp/analysis_model'):
        for file in files:
            if file.startswith('cha_zhi_result_'):
                full_name = os.path.join(root, file)
                data = pd.read_csv(full_name, low_memory=False, dtype={'code': str})
                # 找出列名中包含ratio的列
                ratio_columns = [column for column in data.columns if 'ratio' in column]
                # 其它的列名
                other_columns = [column for column in data.columns if 'chazhi' in column]
                # 从0.8开始，每次增加0.01
                abs_threshold_values = np.arange(0.5, 1, 0.01)
                result = analyze_data(data, ratio_columns, '后续1日最高价利润率')
                data_list.append(result)


def statistic_select_data(data, profit_col_1='后续1日最高价利润率', profit_col_2='后续2日最高价利润率', threshold=1):
    """统计data中指定两个profit列小于给定阈值的数量、天数及其比例,并返回对应的具体选择情况。

    Args:
        data (DataFrame): 输入的数据。
        profit_col_1 (str): 第一个profit列的名称。
        profit_col_2 (str): 第二个profit列的名称。
        threshold (float): 阈值,用于比较profit列。

    Returns:
        dict: 包含统计结果和具体选择情况的字典。
    """
    # 计算第一个profit列相关数据
    profit_1_less_than_threshold = data[data[profit_col_1] < threshold]
    profit_1_less_than_threshold_count = profit_1_less_than_threshold.shape[0]
    profit_1_less_than_threshold_days = profit_1_less_than_threshold['日期'].nunique()
    profit_1_less_than_threshold_count_ratio = profit_1_less_than_threshold_count / data.shape[0] if data.shape[0] != 0 else 0
    profit_1_less_than_threshold_days_ratio = profit_1_less_than_threshold_days / data['日期'].nunique() if data['日期'].nunique() != 0 else 0
    profit_1_less_than_threshold_count_date_list = profit_1_less_than_threshold[['日期', '代码']].apply(lambda x: f"{x['日期']}_{x['代码']}", axis=1).tolist()
    profit_1_less_than_threshold_day_date_list = profit_1_less_than_threshold[['日期', '代码']].drop_duplicates('日期').apply(lambda x: f"{x['日期']}_{x['代码']}", axis=1).tolist()

    # 计算第二个profit列相关数据
    profit_2_less_than_threshold = data[data[profit_col_2] < threshold]
    profit_2_less_than_threshold_count = profit_2_less_than_threshold.shape[0]
    profit_2_less_than_threshold_days = profit_2_less_than_threshold['日期'].nunique()
    profit_2_less_than_threshold_count_ratio = profit_2_less_than_threshold_count / data.shape[0] if data.shape[0] != 0 else 0
    profit_2_less_than_threshold_days_ratio = profit_2_less_than_threshold_days / data['日期'].nunique() if data['日期'].nunique() != 0 else 0
    profit_2_less_than_threshold_count_date_list = profit_2_less_than_threshold[['日期', '代码']].apply(lambda x: f"{x['日期']}_{x['代码']}", axis=1).tolist()
    profit_2_less_than_threshold_day_date_list = profit_2_less_than_threshold[['日期', '代码']].drop_duplicates('日期').apply(lambda x: f"{x['日期']}_{x['代码']}", axis=1).tolist()

    # 计算total_count_date_list
    total_count_date_list = data[['日期', '代码']].apply(lambda x: f"{x['日期']}_{x['代码']}", axis=1).tolist()

    # 组装结果
    result_dict = {
        'day_2_count': profit_2_less_than_threshold_count,
        'day_2_day': profit_2_less_than_threshold_days,
        'day_2_count_ratio': profit_2_less_than_threshold_count_ratio,
        'day_2_day_ratio': profit_2_less_than_threshold_days_ratio,
        'day_2_count_date_list': ','.join(sorted(profit_2_less_than_threshold_count_date_list)),
        'day_2_day_date_list': ','.join(sorted(profit_2_less_than_threshold_day_date_list)),
        'day_1_count': profit_1_less_than_threshold_count,
        'day_1_day': profit_1_less_than_threshold_days,
        'day_1_count_ratio': profit_1_less_than_threshold_count_ratio,
        'day_1_day_ratio': profit_1_less_than_threshold_days_ratio,
        'day_1_count_date_list': ','.join(sorted(profit_1_less_than_threshold_count_date_list)),
        'day_1_day_date_list': ','.join(sorted(profit_1_less_than_threshold_day_date_list)),
        'total_count': data.shape[0],
        'total_days': data['日期'].nunique(),
        'total_count_date_list': ','.join(sorted(total_count_date_list))
    }

    return result_dict

def remove_single_code_days(data):
    """
    Remove rows from the DataFrame where there is only one record for any given 'date'.

    Parameters:
    data (pd.DataFrame): The DataFrame from which to remove rows.

    Returns:
    pd.DataFrame: A DataFrame with the rows removed where only one record exists for each 'date'.
    """
    origin_data = data.copy()
    # 计算每个 'date' 的记录数量
    date_counts = data.groupby('日期').size()

    # 找到那些只有一条记录的 'date'
    single_record_dates = date_counts[date_counts == 1].index

    # 从数据中移除这些日期的记录
    data = origin_data[~origin_data['日期'].isin(single_record_dates)]
    single_record_data = origin_data[origin_data['日期'].isin(single_record_dates)]

    return data, single_record_data

def keep_biggest_day_code(data):
    # 过滤data，相同date的保留最大的count
    # 使用 groupby 和 transform 找到每个 'date' 的最大 'count'
    data = data.copy()
    data['max_count'] = data.groupby('日期')['rf_select_count'].transform('max')
    # 过滤出每个 'date' 中 'count' 等于 'max_count' 的行
    filtered_data = data[data['rf_select_count'] == data['max_count']]
    # 由于可能存在相同 'date' 和相同最大 'count' 的不同行，我们可以选择去除重复的 'date'，保留第一条记录
    filtered_data = filtered_data.drop_duplicates(subset='日期', keep='first')
    return filtered_data

def find_common_and_unique_rows(data1, data2):
    """
    Find common rows and unique rows between two DataFrames based on 'date' and 'code' columns,
    without carrying over the non-key columns from the other DataFrame.

    Parameters:
    data1 (pd.DataFrame): First DataFrame.
    data2 (pd.DataFrame): Second DataFrame.

    Returns:
    tuple of pd.DataFrame: Four DataFrames containing:
        - Rows in data1 that are also in data2
        - Rows in data2 that are also in data1
        - Rows unique to data1
        - Rows unique to data2
    """
    # 获取共有的行
    common_data1 = pd.merge(data1, data2[['date', 'code']], on=['date', 'code'], how='inner')
    common_data2 = pd.merge(data2, data1[['date', 'code']], on=['date', 'code'], how='inner')

    # 获取 data1 独有的行
    unique_data1 = pd.merge(data1, data2[['date', 'code']], on=['date', 'code'], how='left', indicator=True)
    unique_data1 = unique_data1[unique_data1['_merge'] == 'left_only'].drop('_merge', axis=1)

    # 获取 data2 独有的行
    unique_data2 = pd.merge(data2, data1[['date', 'code']], on=['date', 'code'], how='left', indicator=True)
    unique_data2 = unique_data2[unique_data2['_merge'] == 'left_only'].drop('_merge', axis=1)

    return common_data1, common_data2, unique_data1, unique_data2


def analyse_all_select(file_path):
    profit = 1
    result_dict = {}
    # file_path = '../temp/data/all_selected_samples_day2_ratio0.1_profitday1_2023_min_120.csv'
    data = pd.read_csv(file_path, low_memory=False, dtype={'code': str})
    # file_path2 = '../temp/data/all_selected_samples_day2_ratio0.1_profitday2.csv'
    # data2 = pd.read_csv(file_path2, low_memory=False, dtype={'code': str})
    # common_data1, common_data2, unique_data1, unique_data2 = find_common_and_unique_rows(data, data2)
    # data = unique_data2
    remove_single_code_days_data, single_record_dates = remove_single_code_days(data)
    filtered_data = keep_only_single_code_days(data)
    filter_remove_single_code_days_data = keep_only_single_code_days(remove_single_code_days_data)

    filter_remove_single_code_days_data_result = statistic_select_data(filter_remove_single_code_days_data, threshold=profit)
    single_record_dates_result = statistic_select_data(single_record_dates,
                                                                       threshold=profit)
    filtered_data_result = statistic_select_data(filtered_data, threshold=profit)
    data_result = statistic_select_data(data, threshold=profit)
    remove_single_code_days_data_result = statistic_select_data(remove_single_code_days_data, threshold=profit)
    result_dict['filtered_data'] = filtered_data_result
    result_dict['data'] = data_result
    result_dict['remove_single_code_days_data'] = remove_single_code_days_data_result
    result_dict['filter_remove_single_code_days_data'] = filter_remove_single_code_days_data_result
    result_dict['single_record_dates'] = single_record_dates_result
    # 将result_dict写入文件，文件名为'../temp/other/all_selected_samples_day1_ratio0.01_profitday1_result.json'
    base_name = os.path.basename(file_path)
    output_filename = f'../temp/choose_data_result/{base_name}_result.json'
    with open(output_filename, 'w') as file:
        json.dump(result_dict, file)
    # print(filtered_data)

def single_process(result_list,output_filename,all_model_name_dict, profit_key):
    profit_key_2 = profit_key.replace('1', '2')
    final_result_list = []
    # 将result_list按照json_file分组
    first_grouped = pd.DataFrame(result_list).groupby('json_file')
    for json_file, first_group in first_grouped:
        model_name_list = all_model_name_dict[json_file]
        # 只保留all_selected_samples中model_name在model_name_list中的数据
        origin_selected_samples = all_selected_samples[all_selected_samples['model_name'].isin(model_name_list)]
        # 遍历group，获取cha_zhi和min_count
        for index, row in first_group.iterrows():
            # print(f"当前行 {row['json_file']} {row['cha_zhi']} {row['min_count']} {row['thread_day']} {row['select_day_count']}")
            cha_zhi = row['cha_zhi']
            min_count = row['min_count']
            thread_day = row['thread_day']
            # 保留all_selected_samples中model_name包含thread_day的数据
            if thread_day is not None:
                selected_samples = origin_selected_samples[
                    origin_selected_samples['model_name'].str.contains(thread_day)]
            else:
                selected_samples = origin_selected_samples
            # 保留all_selected_samples中cha_zhi大于等于0的数据
            if cha_zhi is not None:
                selected_samples = selected_samples[selected_samples['cha_thread'] >= -cha_zhi]
            grouped_by_date = selected_samples.groupby('日期')
            temp_result_list = []
            for date, group in grouped_by_date:
                grouped = group.groupby('code').agg(max_close=('收盘', 'max'), min_close=('收盘', 'min'),
                                                    current_price=('current_price', 'min'),
                                                    count=('code', 'count'), profit_1=(profit_key, 'mean'), profit_2=(profit_key_2, 'mean'))
                # 输出count大于min_count的数据
                grouped = grouped[grouped['count'] >= min_count]
                grouped = grouped.sort_values('count', ascending=False).head(1)
                if grouped.shape[0] > 0:
                    grouped['date'] = date
                    grouped['code'] = grouped.index
                    temp_result_list.append(grouped)
            temp_all_selected_samples = pd.concat(temp_result_list,
                                                  ignore_index=True) if temp_result_list else pd.DataFrame()
            if temp_all_selected_samples.shape[0] != row['select_day_count']:
                # print(f'当前日期 {row} 代码数量 {temp_all_selected_samples.shape[0]} 选择天数 {row["select_day_count"]} 不一致')
                pass
            # print(temp_all_selected_samples)
            # print(row)
            if temp_all_selected_samples.shape[0] > 0:
                final_result_list.append(temp_all_selected_samples)
                # temp_data = temp_all_selected_samples.groupby(['date', 'code']).agg(count=('code', 'count'), profit=('profit', 'mean'))
                # print(temp_data)
    final_grouped = None
    if final_result_list:
        final_result_list = pd.concat(final_result_list, ignore_index=True)
        # 输出final_result_list不重复的天数
        print(f"共有 {final_result_list['date'].nunique()} 天数据")
        print(f"共有 {final_result_list['code'].nunique()} 条选择")
        # print(final_result_list['code'].value_counts())
        # 将final_result_list按照日期和代码分组，输出对应的数量，还有profit的均值
        final_grouped = final_result_list.groupby(['date', 'code']).agg(count=('code', 'count'),
                                                                        profit_1=('profit_1', 'mean'), profit_2=('profit_2', 'mean'))
        # print(final_grouped)
        # 将索引 'date' 和 'code' 重置为普通列
        final_grouped_reset = final_grouped.reset_index()
        # print(final_result_list)
        # 将min_count，base_name，ratio，profit_key拼接到文件名中

        final_grouped_reset.to_csv(output_filename, index=False)
        analyse_all_select(output_filename)


def process_task(args):
    ratio, count_min_count, param_file_path, profit_key, all_model_name_dict, result_list = args
    base_name = os.path.basename(param_file_path)
    output_filename = f'../temp/choose_data/all_{base_name}_min_{count_min_count}_ratio_{ratio}_profit{profit_key}.csv'

    if os.path.exists(output_filename):
        print(f"文件 {output_filename} 已存在")
        analyse_all_select(output_filename)
        return

    # 过滤 result_list
    filtered_results = [result for result in result_list if
                        result['bad_count'] / result['select_day_count'] <= ratio and result[
                            'select_day_count'] >= count_min_count]
    print(
        f"共有 {len(filtered_results)} 个模型满足条件 筛选条件为 bad_count/total_count小于等于{ratio} select_day_count大于等于{count_min_count} {param_file_path} {profit_key}")

    if len(filtered_results) == 0:
        return

    single_process(filtered_results, output_filename, all_model_name_dict, profit_key)

def get_result_select(all_selected_samples, param_result_list, all_model_name_dict, profit_key='后续1日最高价利润率', profit_key_2='后续2日最高价利润率'):
    """
    针对all_selected_samples进行param_result_list的选择，返回满足要求的数据
    :param all_selected_samples:
    :param param_result_list:
    :param all_model_name_dict:
    :param profit_key:
    :param profit_key_2:
    :return:
    """
    final_result_list = []
    # 将result_list按照json_file分组
    first_grouped = pd.DataFrame(param_result_list).groupby('json_file')
    for json_file, first_group in first_grouped:
        model_name_list = all_model_name_dict[json_file]
        # 只保留all_selected_samples中model_name在model_name_list中的数据
        origin_selected_samples = all_selected_samples[all_selected_samples['model_name'].isin(model_name_list)]
        # 遍历group，获取cha_zhi和min_count
        for index, row in first_group.iterrows():
            # print(f"当前行 {row['json_file']} {row['cha_zhi']} {row['min_count']} {row['thread_day']} {row['select_day_count']}")
            cha_zhi = row['cha_zhi']
            min_count = row['min_count']
            thread_day = row['thread_day']
            # 保留all_selected_samples中model_name包含thread_day的数据
            if thread_day is not None:
                selected_samples = origin_selected_samples[
                    origin_selected_samples['model_name'].str.contains(thread_day)]
            else:
                selected_samples = origin_selected_samples
            # 保留all_selected_samples中cha_zhi大于等于0的数据
            if cha_zhi is not None:
                selected_samples = selected_samples[selected_samples['cha_thread'] >= -cha_zhi]
            grouped_by_date = selected_samples.groupby('日期')
            temp_result_list = []
            for date, group in grouped_by_date:
                grouped = group.groupby('code').agg(max_close=('收盘', 'max'), min_close=('收盘', 'min'),
                                                    current_price=('current_price', 'min'),
                                                    count=('code', 'count'), profit_1=(profit_key, 'mean'),
                                                    profit_2=(profit_key_2, 'mean'))
                # 输出count大于min_count的数据
                grouped = grouped[grouped['count'] >= min_count]
                grouped = grouped.sort_values('count', ascending=False).head(1)
                if grouped.shape[0] > 0:
                    grouped['date'] = date
                    grouped['code'] = grouped.index
                    grouped['cha_zhi'] = cha_zhi
                    grouped['min_count'] = min_count
                    grouped['thread_day'] = thread_day
                    grouped['select_day_count'] = row['select_day_count']
                    grouped['bad_count'] = row['bad_count']
                    grouped['json_file'] = json_file
                    temp_result_list.append(grouped)
            temp_all_selected_samples = pd.concat(temp_result_list,
                                                  ignore_index=True) if temp_result_list else pd.DataFrame()
            if temp_all_selected_samples.shape[0] != row['select_day_count']:
                # print(f'当前日期 {row} 代码数量 {temp_all_selected_samples.shape[0]} 选择天数 {row["select_day_count"]} 不一致')
                pass
            # print(temp_all_selected_samples)
            # print(row)
            if temp_all_selected_samples.shape[0] > 0:
                final_result_list.append(temp_all_selected_samples)
                # temp_data = temp_all_selected_samples.groupby(['date', 'code']).agg(count=('code', 'count'), profit=('profit', 'mean'))
                # print(temp_data)
    all_selected_samples_with_param = None
    if final_result_list:
        all_selected_samples_with_param = pd.concat(final_result_list, ignore_index=True)
    return all_selected_samples_with_param


def process_subset(args):
    """
    处理分割的result_list部分
    """
    all_selected_samples, subset_result_list, all_model_name_dict, profit_key = args
    return get_result_select(all_selected_samples, subset_result_list, all_model_name_dict, profit_key)

def get_all_param_select(file_path='../temp/data/all_selected_samples_20240102_20240425.csv', profit_key='后续1日最高价利润率', param_file_path='../final_zuhe/other/result_list_day1_2023filter.json'):
    """
    获取所有参数的选择，通过并行处理result_list加速
    """
    all_selected_samples = pd.read_csv(file_path, low_memory=False, dtype={'代码': str})
    all_selected_samples['日期'] = pd.to_datetime(all_selected_samples['日期'])
    # all_selected_samples = all_selected_samples[all_selected_samples['日期'] < '2024-04-24']
    all_selected_samples['code'] = all_selected_samples['代码']
    if 'current_price' not in all_selected_samples.columns:
        all_selected_samples['current_price'] = all_selected_samples['收盘']
    profit_key_2 = profit_key.replace('1', '2')
    if profit_key not in all_selected_samples.columns:
        all_selected_samples[profit_key] = 0
    if profit_key_2 not in all_selected_samples.columns:
        all_selected_samples[profit_key_2] = 0

    with open(param_file_path, 'r') as file:
        result_list = json.load(file)
    file_list = ['good_all_model_reports_cuml_100_200_thread12.json',
                 'good_all_model_reports_cuml_200_200_thread12.json',
                 'good_all_model_reports_cuml_300_200_thread12.json',
                 'good_all_model_reports_cuml_100_200_thread2.json', 'good_all_model_reports_cuml_200_200_thread2.json',
                 'good_all_model_reports_cuml_300_200_thread2.json',
                 'good_all_model_reports_cuml_100_200_thread1.json', 'good_all_model_reports_cuml_200_200_thread1.json',
                 'good_all_model_reports_cuml_300_200_thread1.json']
    all_model_name_dict = {}
    for file_str in file_list:
        with open(f'../final_zuhe/other/{file_str}', 'r') as file:
            model_info_list = json.load(file)
            model_name_list = [model_info['model_name'] for model_info in model_info_list]
            all_model_name_dict[file_str] = model_name_list

    # 分割result_list
    num_processes = 25  # 或者根据你的CPU核心数
    print(f"共有 {len(result_list)} 个参数")
    chunk_size = len(result_list) // num_processes
    result_list_chunks = [result_list[i:i + chunk_size] for i in range(0, len(result_list), chunk_size)]

    # 使用多进程处理数据
    with Pool(num_processes) as pool:
        results = pool.map(process_subset, [(all_selected_samples, chunk, all_model_name_dict, profit_key) for chunk in result_list_chunks])

    # 合并结果
    all_selected_samples_with_param = pd.concat(results)

    print(f'第一层命中数量{all_selected_samples_with_param.shape[0]}')
    # 将all_selected_samples_with_param写入文件
    base_name = os.path.basename(file_path)
    param_base_name = os.path.basename(param_file_path)
    output_filename = f'../temp/back/all_{base_name}_param_{param_base_name}.csv'
    all_selected_samples_with_param.to_csv(output_filename, index=False)
    return output_filename

def choose_code_from_all_selected_samples(all_selected_samples, profit_key='后续1日最高价利润率', param_file_path='../final_zuhe/other/result_list_day1_2023filter.json'):
    # param_file_path = '../final_zuhe/other/result_list_day1_2023.json'
    # profit_key = '后续1日最高价利润率'
    all_selected_samples['code'] = all_selected_samples['代码']
    if 'current_price' not in all_selected_samples.columns:
        all_selected_samples['current_price'] = all_selected_samples['收盘']

    if profit_key not in all_selected_samples.columns:
        all_selected_samples[profit_key] = 0

    with open(param_file_path, 'r') as file:
        result_list = json.load(file)
    # 获取result_list中最大的select_day_count
    max_select_day_count = max([result['select_day_count'] for result in result_list])
    # 获取result_list中的不重复的json_file
    json_file_list = list(set([result['json_file'] for result in result_list]))

    ratio_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,0.16, 0.17, 0.18, 0.19, 0.2]
    # 将ratio_list逆序排列
    ratio_list = ratio_list[::-1]
    current_count = 0
    count_min_count_list = []
    while current_count < max_select_day_count:
        count_min_count_list.append(current_count)
        current_count += 10
    file_list = ['good_all_model_reports_cuml_100_200_thread12.json',
                 'good_all_model_reports_cuml_200_200_thread12.json',
                 'good_all_model_reports_cuml_300_200_thread12.json',
                 'good_all_model_reports_cuml_100_200_thread2.json', 'good_all_model_reports_cuml_200_200_thread2.json',
                 'good_all_model_reports_cuml_300_200_thread2.json',
                 'good_all_model_reports_cuml_100_200_thread1.json', 'good_all_model_reports_cuml_200_200_thread1.json',
                 'good_all_model_reports_cuml_300_200_thread1.json']

    all_model_name_dict = {}
    for file_str in file_list:
        with open(f'../final_zuhe/other/{file_str}', 'r') as file:
            model_info_list = json.load(file)
            model_name_list = [model_info['model_name'] for model_info in model_info_list]
            all_model_name_dict[file_str] = model_name_list

    tasks = [(ratio, count_min_count, param_file_path, profit_key, all_model_name_dict, result_list) for ratio in
             ratio_list for count_min_count in count_min_count_list]
    print(f"共有 {len(tasks)} 个任务")
    # 将tasks按照ratio降序排列
    tasks = sorted(tasks, key=lambda x: x[0], reverse=True)

    # Create a pool of processes
    with Pool(20) as pool:
        pool.map(process_task, tasks)


def split_list(count_list, split_count):
    max_count = len(set(count_list))
    if max_count < split_count:
        return list(set(count_list))
    count_list = list(set(count_list))
    sorted_count_list = sorted(count_list)
    total_count = len(sorted_count_list)
    target_size = total_count // split_count

    split_points = []
    current_index = target_size

    for i in range(split_count - 1):
        split_points.append(sorted_count_list[current_index])
        current_index += target_size

    return split_points

def get_detail_analysis(group_select):
    """
    获取group_select的详细分析,主要是进行（选择最高的 选择大于min_count的 选择单数据的 选择多数据的）的结果
    :param group_select:
    :return:
    """
    result_dict = {}
    # 移除同一天只有一个代码的数据
    remove_single_code_days_data, single_record_dates = remove_single_code_days(group_select)
    biggest_data = keep_biggest_day_code(group_select)
    biggest_remove_single_code_days_data = keep_biggest_day_code(remove_single_code_days_data)
    result_dict['data'] = statistic_select_data(group_select)
    result_dict['remove_single_code_days_data'] = statistic_select_data(remove_single_code_days_data)
    result_dict['biggest_data'] = statistic_select_data(biggest_data)
    result_dict['biggest_remove_single_code_days_data'] = statistic_select_data(biggest_remove_single_code_days_data)
    result_dict['single_record_dates'] = statistic_select_data(single_record_dates)
    # 将result_dict转换为DataFrame
    rows_list = []
    for strategy, metrics in result_dict.items():
        # 创建一个新的字典，包含文件名、策略和所有度量
        row = {'strategy': strategy}
        row.update(metrics)
        rows_list.append(row)
    if len(rows_list) > 0:
        result_df = pd.DataFrame(rows_list)
    else:
        result_df = pd.DataFrame()
    return result_df

def filter_duplicated_date_code(data):
    """
    过滤data中的重复日期和代码
    :param data:
    :return:
    """
    data = data.copy()
    data = data.drop_duplicates(subset=['日期', '代码'])
    return data

def analyse_first_min_count_select(all_selected_samples, output_filename_path, param_dict):
    """
    分析出第一层的选择数据，all_selected_samples为模型组 cha date_count筛选后的数据，返回结果会加上min_count限制和选择的方法限制（选择最高的 选择大于min_count的 选择单数据的 选择多数据的）
    :param all_selected_samples:
    :return:
    """
    if all_selected_samples.shape[0] == 0:
        print(f'all_selected_samples is empty {all_selected_samples}')
        return pd.DataFrame()
    result_list = []
    origin_grouped = all_selected_samples.groupby(['日期', '代码']).agg(rf_select_count=('代码', 'count'), min_close=('收盘', 'first'), 后续1日最高价利润率=('后续1日最高价利润率', 'mean'), 后续2日最高价利润率=('后续2日最高价利润率', 'mean'))
    origin_grouped = origin_grouped.reset_index()
    # 获取origin_grouped所有的数量
    count_list = origin_grouped['rf_select_count'].tolist()
    min_count_list = list(set(count_list))
    min_count_list = split_list(min_count_list, 100)
    min_count_list.sort()
    # 如果output_filename_path存在，直接返回
    if os.path.exists(output_filename_path):
        return
    # 判断output_filename_path路径是否存在，不存在则创建
    if not os.path.exists(output_filename_path):
        os.makedirs(output_filename_path)
    min_output_filename_path = os.path.join(output_filename_path, 'min_count.csv')

    for min_count in min_count_list:
        # print(f"开始分析第一层参数min_count 当前时间{datetime.now()} min_count{min_count} {origin_grouped.shape[0]}")
        # 筛选出count大于min_count的数据
        selected_samples = origin_grouped[origin_grouped['rf_select_count'] >= min_count]
        more_result_df = get_detail_analysis(selected_samples)
        more_result_df['min_count'] = min_count
        more_result_df['is_more_min_count'] = True
        result_list.append(more_result_df)

        selected_samples = origin_grouped[origin_grouped['rf_select_count'] < min_count]
        less_result_df = get_detail_analysis(selected_samples)
        less_result_df['min_count'] = min_count
        less_result_df['is_more_min_count'] = False
        result_list.append(less_result_df)
        # print(f"完成分析第一层参数min_count 当前时间{datetime.now()} min_count{min_count} {origin_grouped.shape[0]}")
    if len(result_list) > 0:
        result_df = pd.concat(result_list)
    else:
        result_df = pd.DataFrame()
    # 删除total_count为0的数据
    if 'total_count' in result_df.columns:
        result_df = remove_duplicate_rows(result_df)
        result_df = result_df[result_df['total_days'] > 3]
        result_df = result_df[result_df['day_2_day_ratio'] <= 0.1]
        # 遍历param_dict，将param_dict的值添加到result_df中
        for key, value in param_dict.items():
            result_df[key] = value
        if result_df.shape[0] > 0:
            result_df.to_csv(min_output_filename_path, index=False)
    else:
        print(f'total_count not in columns{result_df} {all_selected_samples}')

def remove_duplicate_rows(df):
    """
    删除DataFrame中的重复行,根据包含"date_list"的列的值来判断是否相同。

    Args:
        df (DataFrame): 输入的DataFrame。

    Returns:
        DataFrame: 删除重复行后的DataFrame。
    """
    # 找出包含"date_list"的列
    date_list_columns = [col for col in df.columns if 'date_list' in col]

    # 如果没有包含"date_list"的列,则直接返回原始DataFrame
    if not date_list_columns:
        return df

    # 根据包含"date_list"的列删除重复行
    df_filtered = df.drop_duplicates(subset=date_list_columns)
    print(f"过滤前 {df.shape[0]} 删除重复行后的DataFrame大小: {df_filtered.shape[0]}")
    return df_filtered

def analyse_first_cha_zhi_select(origin_selected_samples, output_filename_path, param_dict):
    cha_zhi_result_list = []
    # 获取origin_selected_samples的所有不重复cha_thread，保留两位小数
    cha_thread_list = origin_selected_samples['cha_thread'].tolist()
    cha_thread_list = [round(cha_thread, 2) for cha_thread in cha_thread_list]
    cha_thread_list = list(set(cha_thread_list))
    cha_thread_list.sort()
    cha_zhi_list = cha_thread_list
    cha_zhi_list.append(None)
    cha_zhi_list.reverse()

    # 将all_selected_samples按照日期分组
    for cha_zhi in cha_zhi_list:

        # 保留all_selected_samples中cha_zhi大于等于0的数据
        if cha_zhi is not None:
            more_selected_samples = origin_selected_samples[origin_selected_samples['cha_thread'] >= cha_zhi]
            less_selected_samples = origin_selected_samples[origin_selected_samples['cha_thread'] < cha_zhi]
        else:
            # 只保留每天cha_thread最大的数据
            more_selected_samples = origin_selected_samples.loc[origin_selected_samples.groupby('日期')['cha_thread'].idxmax()]
            less_selected_samples = origin_selected_samples.loc[origin_selected_samples.groupby('日期')['cha_thread'].idxmin()]
        print(
            f"开始分析第一层参数cha_zhi 当前时间{datetime.now()} cha_zhi {cha_zhi} more_selected_samples {more_selected_samples.shape[0]} less_selected_samples {less_selected_samples.shape[0]}")
        cha_output_filename_path = os.path.join(output_filename_path, str(cha_zhi) + '_more')
        param_dict['cha_zhi'] = cha_zhi
        param_dict['is_cha_zhi_more'] = True
        more_result_df = analyse_first_min_count_select(more_selected_samples, cha_output_filename_path, param_dict)
        cha_output_filename_path = os.path.join(output_filename_path, str(cha_zhi) + '_less')
        param_dict['is_cha_zhi_more'] = False
        less_result_df = analyse_first_min_count_select(less_selected_samples, cha_output_filename_path, param_dict)
        print(f"完成分析第一层参数cha_zhi 当前时间{datetime.now()} cha_zhi {cha_zhi} {origin_selected_samples.shape[0]} more_selected_samples {more_selected_samples.shape[0]} less_selected_samples {less_selected_samples.shape[0]} cha_zhi进度{cha_zhi_list.index(cha_zhi)}/{len(cha_zhi_list)}")


def analyse_first_date_count_select_parallel(date_count, origin_selected_samples, model_info_list, output_filename_path, param_dict):
    more_model_name_list = [model_info['model_name'] for model_info in model_info_list if
                            model_info['date_count'] >= date_count]
    more_origin_selected_samples = origin_selected_samples[origin_selected_samples['model_name'].isin(more_model_name_list)]
    less_model_name_list = [model_info['model_name'] for model_info in model_info_list if
                            model_info['date_count'] < date_count]
    less_origin_selected_samples = origin_selected_samples[origin_selected_samples['model_name'].isin(less_model_name_list)]
    print(
        f"开始分析第一层参数date_count 当前时间{datetime.now()} date_count {date_count} more_origin_selected_samples {more_origin_selected_samples.shape[0]} less_origin_selected_samples {less_origin_selected_samples.shape[0]}")
    date_output_filename_path = os.path.join(output_filename_path, str(date_count) + '_more')
    param_dict['date_count'] = date_count
    param_dict['is_date_count_more'] = True
    more_result_df = analyse_first_cha_zhi_select(more_origin_selected_samples, date_output_filename_path, param_dict)
    date_output_filename_path = os.path.join(output_filename_path, str(date_count) + '_less')
    param_dict['is_date_count_more'] = False
    less_result_df = analyse_first_cha_zhi_select(less_origin_selected_samples, date_output_filename_path, param_dict)
    return more_result_df, less_result_df

def analyse_first_date_count_select(origin_selected_samples, model_info_list, output_filename_path, param_dict, processes=25):
    """
    分析出第一层的选择数据,all_selected_samples为模型组 cha date_count筛选后的数据,返回结果会加上min_count限制和选择的方法限制(选择最高的 选择大于min_count的 选择单数据的 选择多数据的)
    :param origin_selected_samples:
    :param model_info_list:
    :return:
    """
    date_count_list = [model_info['date_count'] for model_info in model_info_list]
    date_count_split_list = list(set(date_count_list))
    date_count_split_list = split_list(date_count_split_list, 100)
    date_count_split_list.sort()
    print(f"开始分析第一层参数date_count 当前时间{datetime.now()} {date_count_split_list}")
    pool = Pool(processes=processes)
    results = []
    for date_count in date_count_split_list:
        result = pool.apply_async(analyse_first_date_count_select_parallel,
                                  args=(date_count, origin_selected_samples, model_info_list, output_filename_path, param_dict))
        results.append(result)

    pool.close()
    pool.join()

def save_all_selected_samples(file_path, processes=25):
    """
    第一次好参数性能获取，通过带有rf模型的select数据获取第一层参数的性能
    :param all_selected_samples:
    :param min_count:
    :return:
    """
    # 获取file_path的最后一层文件名
    base_name = os.path.basename(file_path)
    # 获取file_path倒数第二层文件名
    param_base_name = os.path.basename(os.path.dirname(file_path))


    all_selected_samples = low_memory_load(file_path)
    all_selected_samples['日期'] = pd.to_datetime(all_selected_samples['日期'])
    # 将日期转换为字符串保留年月日
    all_selected_samples['日期'] = all_selected_samples['日期'].dt.strftime('%Y-%m-%d')
    all_model_name_dict = get_all_model_list()
    if '后续2日最高价利润率' not in all_selected_samples.columns:
        all_selected_samples['后续2日最高价利润率'] = 0
    if '后续1日最高价利润率' not in all_selected_samples.columns:
        all_selected_samples['后续1日最高价利润率'] = 0
    for json_file, model_info_list in all_model_name_dict.items():
        output_filename_path = f'../final_zuhe/first_param/{param_base_name}/{base_name}/{json_file}'
        param_dict = {'json_file': json_file}

        print(f"开始模型组第一层参数遍历 当前时间{datetime.now()} json_file {json_file} model_info_list {len(model_info_list)} {file_path}")
        result_df = analyse_first_date_count_select(all_selected_samples, model_info_list, output_filename_path, param_dict, processes=processes)
        print(f"完成模型组第一层参数遍历 当前时间{datetime.now()} json_file{json_file} model_info_list{len(model_info_list)} {file_path} 模型组进度{list(all_model_name_dict.keys()).index(json_file)}/{len(all_model_name_dict)}")
        # if len(first_param_result_list) > 0:
        #     first_param_result_df = pd.concat(first_param_result_list)
        #     # first_param_result_df = remove_duplicate_rows(first_param_result_df)
        #     # 删除data中列名包含date_list的列
        #     # first_param_result_df = first_param_result_df.loc[:, ~first_param_result_df.columns.str.contains('date_list')]
        # else:
        #     first_param_result_df = pd.DataFrame()
        # 过滤first_param_result_df，只保留day_2_count_ratio小于0.2的数据
        # first_param_result_df = first_param_result_df[first_param_result_df['day_2_count_ratio'] <= 0.2]
        # first_param_result_df.to_csv(output_filename, index=False)
        # print(f"保存文件 {output_filename} len {first_param_result_df.shape[0]}")


def summarize_quantities(file_path, bad_count):
    from collections import defaultdict
    import csv

    # 创建一个字典来存储每个编号的累积数量和其他数据
    data_summary = defaultdict(lambda: {'price': 0, 'quantity': 0, 'value': 0})

    # 读取文件
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # 确保行非空
                row = [item.strip() for item in row]
                # 解析行数据
                identifier = row[0]
                price = float(row[1])
                quantity = float(row[2])
                value = float(row[3])

                # 更新字典
                if identifier in data_summary:
                    data_summary[identifier]['quantity'] += quantity
                else:
                    data_summary[identifier] = {'price': price, 'quantity': quantity, 'value': value}

    # 对字典按数量排序
    sorted_data = sorted(data_summary.items(), key=lambda x: x[1]['quantity'], reverse=True)
    # 计算总数量
    total_quantity = sum(info['quantity'] for identifier, info in sorted_data)
    code_len = len(sorted_data)
    result = []
    # 输出结果
    for identifier, info in sorted_data:
        # 获取file_path的日期
        date = file_path.split('/')[-1].split('_')[0]
        line = f"date：{date}  ID: {identifier:6}, Price: {info['price']:8.2f}, Quantity: {info['quantity']:9.2f}, total:{total_quantity:6}  code_len:{code_len:6}  Value: {info['value']:8.2f}"

        if info['value'] < 1:
            # print(line)
            bad_count += 1
        result.append(line)

        break
    return result, bad_count


def sort_all_select():
    file_name_list = []
    for root, dirs, files in os.walk('../final_zuhe/select'):
        for file in files:
            if'real_time_good_price' in file:
                full_name = os.path.join(root, file)
                file_name_list.append(full_name)
    result_dict = {}
    bad_count = 0
    for file_name in file_name_list:
        sorted_data, bad_count = summarize_quantities(file_name, bad_count)
        key = file_name.split('/')[-1].split('.')[0]
        if sorted_data:
            result_dict[key] = sorted_data
    # 将result_dict写入文件
    with open('../final_zuhe/other/sorted_all_select.json', 'w') as file:
        json.dump(result_dict, file)
    select_day_count = len(result_dict)
    return bad_count, select_day_count


def merge_and_sort_json_files(file_name_list):
    """
    Merge JSON files and sort the merged result by 'total_days' in descending order
    and then by 'profit_less_than_1_days_ratio' in ascending order, including the base filename
    in the DataFrame to trace back to the original file.

    Parameters:
    file_name_list (list of str): List containing filenames of JSON files.

    Returns:
    pd.DataFrame: A DataFrame sorted according to the specified criteria, with an additional
                  column for the file base name.
    """
    data_list = []  # Store data from all files

    # Loop through each file name provided in the list
    for file_name in file_name_list:
        with open(file_name, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error reading file: {file_name}")
                continue
            'all_result_list_day1_2023filter.json_min_0_ratio_0.01_profit后续1日最高价利润率.csv_result.json'
            # Extract each part of the JSON and append it to the list with an identifier
            for key, value in data.items():
                value['category'] = key
                value['file_base_name'] = os.path.basename(file_name)  # Add base file name

                data_list.append(value)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)

    # Sort the DataFrame first by 'total_days' descending, then by 'profit_less_than_1_days_ratio' ascending
    # sorted_df = df.sort_values(by=['total_days', 'profit_less_than_1_days_ratio'], ascending=[False, True])

    return df

def sort_choose_data_result():
    # 遍历../temp/choose_data_result目录下的所有文件
    file_name_list = []
    for root, dirs, files in os.walk('../temp/choose_data_result'):
        for file in files:
            if file.endswith('.json') and 'result_list_day2_2023filter.json' in file:
                full_name = os.path.join(root, file)
                file_name_list.append(full_name)
    sorted_df = merge_and_sort_json_files(file_name_list)
    print(sorted_df)

def analyse_select(data):
    profit = 1
    result_dict = {}
    remove_single_code_days_data, single_record_dates = remove_single_code_days(data)
    filtered_data = keep_biggest_day_code(data)
    filter_remove_single_code_days_data = keep_biggest_day_code(remove_single_code_days_data)

    filter_remove_single_code_days_data_result = statistic_select_data(filter_remove_single_code_days_data, threshold=profit)
    single_record_dates_result = statistic_select_data(single_record_dates,
                                                                       threshold=profit)
    filtered_data_result = statistic_select_data(filtered_data, threshold=profit)
    data_result = statistic_select_data(data, threshold=profit)
    remove_single_code_days_data_result = statistic_select_data(remove_single_code_days_data, threshold=profit)
    result_dict['filtered_data'] = filtered_data_result
    result_dict['data'] = data_result
    result_dict['remove_single_code_days_data'] = remove_single_code_days_data_result
    result_dict['filter_remove_single_code_days_data'] = filter_remove_single_code_days_data_result
    result_dict['single_record_dates'] = single_record_dates_result
    # 将result_dict转换为DataFrame
    rows_list = []
    for strategy, metrics in result_dict.items():
        # 创建一个新的字典，包含文件名、策略和所有度量
        row = {'strategy': strategy}
        row.update(metrics)
        rows_list.append(row)
    if len(rows_list) > 0:
        result_df = pd.DataFrame(rows_list)
    else:
        result_df = pd.DataFrame()
    return result_df

def parse_filename(filename):
    # 移除文件扩展名
    base = filename[:-5]  # 去掉'.json'

    # 按照下划线分割基础字符串
    parts = base.split('_')

    # 创建字典来存放解析的组件
    parsed_dict = {}

    # 遍历parts列表，根据关键词提取值
    i = 0
    while i < len(parts):
        if parts[i] == 'ratio':
            parsed_dict['ratio'] = parts[i + 1]
            i += 2
        elif parts[i] == 'thread' and parts[i + 1] == 'day':
            parsed_dict['thread_day'] = parts[i + 2]
            i += 3
        elif parts[i] == 'min' and parts[i + 1] == 'day' and parts[i + 2] == 'count':
            parsed_dict['min_day_count'] = parts[i + 3]
            i += 4
        elif parts[i] == 'min' and parts[i + 1] == 'select' and parts[i + 2] == 'count':
            parsed_dict['min_select_count'] = parts[i + 3]
            i += 4
        elif parts[i] == 'json' and parts[i + 1] == 'file':
            # 提取json_file部分，它可能包含多个下划线连接的部分
            json_file_parts = []
            i += 2
            while i < len(parts) and parts[i] != 'thread':
                json_file_parts.append(parts[i])
                i += 1
            parsed_dict['json_file'] = '_'.join(json_file_parts)
        elif parts[i] == 'thread':
            # 提取thread部分，它可能包含数字
            parsed_dict['thread'] = parts[i + 1]
            i += 2

    return parsed_dict

def filter_data(data, args):
    data = data.copy()
    ratio, thread_day, min_day_count, min_select_count, json_file, strategy = args
    # ratio = 0.2
    # min_day_count = 20
    # min_select_count = 0
    # json_file = 'both'
    # thread_day = 'both'
    # 保留data中ratio小于等于ratio的数据
    data = data[data['ratio'] <= ratio]
    # 保留data中thread_day等于thread_day的数据,如果thread_day为'both'则保留所有数据
    if thread_day != 'both':
        data = data[data['thread_day'] == thread_day]
    if json_file != 'both':
        data = data[data['json_file'] == json_file]
    # 保留data中select_day_count大于等于min_day_count的数据
    data = data[data['select_day_count'] >= min_day_count]
    # 合并相同date和code的数据，其他列都取第一个值，同时增加一个select_count列，记录相同date和code的数量
    data = data.groupby(['date', 'code']).agg(select_count=('code', 'count'), profit_1=('profit_1', 'first'), profit_2=('profit_2', 'first'), current_price=('current_price', 'first'))
    data = data.reset_index()
    # 选出select_count大于等于min_select_count的数据
    data = data[data['select_count'] >= min_select_count]
    remove_single_code_days_data, single_record_dates = remove_single_code_days(data)
    filtered_data = keep_only_single_code_days(data)
    filter_remove_single_code_days_data = keep_only_single_code_days(remove_single_code_days_data)
    if strategy == 'filtered_data':
        return filtered_data
    elif strategy == 'data':
        return data
    elif strategy == 'remove_single_code_days_data':
        return remove_single_code_days_data
    elif strategy == 'filter_remove_single_code_days_data':
        return filter_remove_single_code_days_data
    elif strategy == 'single_record_dates':
        return single_record_dates
    return data

def filter_select_data(data, args):
    # print(f"开始处理任务 {args}")
    ratio, thread_day, min_day_count, min_select_count, json_file = args
    data = data[data['ratio'] <= ratio]
    # 保留data中thread_day等于thread_day的数据,如果thread_day为'both'则保留所有数据
    if thread_day != 'both':
        data = data[data['thread_day'] == thread_day]
    if json_file != 'both':
        data = data[data['json_file'] == json_file]
    # 保留data中select_day_count大于等于min_day_count的数据
    data = data[data['select_day_count'] >= min_day_count]
    # 合并相同date和code的数据，其他列都取第一个值，同时增加一个select_count列，记录相同date和code的数量
    data = data.groupby(['date', 'code']).agg(select_count=('code', 'count'), profit_1=('profit_1', 'first'),
                                              profit_2=('profit_2', 'first'), current_price=('current_price', 'first'))
    data = data.reset_index()
    # 如果data为空，则返回空字典
    if data.shape[0] == 0:
        # 返回空DataFrame
        return pd.DataFrame()
    # 在data中恢复date和code列
    data = data.reset_index()
    result_df = analyse_select(data)
    if result_df.empty:
        return result_df
    result_df['ratio'] = ratio
    result_df['thread_day'] = thread_day
    result_df['min_day_count'] = min_day_count
    result_df['min_select_count'] = min_select_count
    result_df['json_file'] = json_file
    return result_df


def process_good_param_task(data, tasks):
    """
    处理单个任务
    """
    print(f"开始处理任务")
    results = []
    for task in tasks:
        temp_df = filter_select_data(data, task)
        if not temp_df.empty:
            results.append(temp_df)
    if len(results) > 0:
        return pd.concat(results, ignore_index=True)
    return None

def chunkify(lst, n):
    """将列表分割成n个近似等长的子列表"""
    return [lst[i::n] for i in range(n)]

def get_good_param_by_param_select(file_path=f'../temp/back/all_all_selected_samples_20240102_20240425.csv_param_result_list_day1_2023filter.json.csv'):
    """
    二次好参数性能获取，通过带有第一层参数的select数据获取第二层参数的性能
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path, low_memory=False)
    data['ratio'] = data['bad_count'] / data['select_day_count']
    ratio_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                  0.17, 0.18, 0.19, 0.2]  # 失败率
    # 获取data中不重复的thread_day
    thread_day_list = list(data['thread_day'].unique())
    thread_day_list.append('both')
    # 获取data中最大的select_day_count
    max_select_day_count = max(data['select_day_count'])
    min_day_count_list = []
    current_count = 0
    while current_count < max_select_day_count:
        min_day_count_list.append(current_count)
        current_count += 10
    min_select_count_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500, 600] # 最小命中规则的数量
    # 获取data中不重复的json_file
    json_file_list = list(data['json_file'].unique())
    json_file_list.append('both')
    # min_select_count_list = [0]
    # thread_day_list = ['both']
    # json_file_list = ['both']
    # 生成所有的参数组合
    tasks = [(ratio, thread_day, min_day_count, min_select_count, json_file) for ratio in ratio_list for thread_day in thread_day_list for min_day_count in min_day_count_list for min_select_count in min_select_count_list for json_file in json_file_list]
    print(f"共有 {len(tasks)} 个任务")
    # 这里设置你希望的进程数，通常不超过你机器的核心数
    num_processes = 20
    # 截取tasks的前1000个任务
    # tasks = tasks[:10000]
    # 将任务列表分割成多个子列表，每个进程处理一个子列表
    tasks_chunks = chunkify(tasks, num_processes * 200)

    # 创建进程池并处理数据
    with Pool(num_processes) as pool:
        # 使用 functools.partial 来绑定 data 参数
        func = partial(process_good_param_task, data)
        all_results = pool.map(func, tasks_chunks)

    result_list = [result for result in all_results if result is not None]
    if len(result_list) > 0:
        result_df = pd.concat(result_list, ignore_index=True)
        # 将result_df写入文件
        base_name = os.path.basename(file_path)
        output_filename = f'../temp/back/good_param_{base_name}.csv'
        result_df.to_csv(output_filename, index=False)

def filter_good_param(file_path = '../temp/back/good_param_all_all_selected_samples_20240102_20240425.csv_param_result_list_day1_2023filter.json.csv.csv', day_2_day_ratio=0.1, day_1_day_ratio=0.1):
    data = pd.read_csv(file_path, low_memory=False)
    # 筛选出day_2_day_ratio小于0.1的数据
    data = data[data['day_2_day_ratio'] <= day_2_day_ratio]
    # 筛选出day_1_day_ratio小于0.1的数据
    data = data[data['day_1_day_ratio'] <= day_1_day_ratio]
    # 去除total_days为0的数据
    data = data[data['total_days'] != 0]
    # 按照total_days降序排列
    data = data.sort_values(by='total_days', ascending=False)
    return data


def process_param_batch(data, tasks):
    """
    处理一个任务批次，返回过滤后数据的列表
    """
    batch_results = []
    for task in tasks:
        temp_df = filter_data(data, task)
        if not temp_df.empty:
            batch_results.append(temp_df)
    return batch_results


def chunk_tasks(tasks, chunk_size):
    """
    将任务列表分成多个批次
    """
    for i in range(0, len(tasks), chunk_size):
        yield tasks[i:i + chunk_size]

def good_param_first_select(data):
    """
    最终通过好的一次参数选股
    :param data:
    :return:
    """
    result_list = []
    bad_ratio_list = [0.0, 0.1]
    for bad_ratio in bad_ratio_list:
        # 筛选出bad_count小于1的数据
        result_df = data[(data['bad_count'] / data['select_day_count']) <= bad_ratio]
        result_df = result_df.groupby(['date', 'code']).agg(select_count=('code', 'count'), profit_1=('profit_1', 'first'),
                                                            profit_2=('profit_2', 'first'), current_price=('current_price', 'first'))
        result_df = result_df.reset_index()
        result_df['ratio'] = bad_ratio
        print(f"{bad_ratio}共有 {result_df.shape[0]} 个选股")
        print(result_df)
        result_list.append(result_df)
    if len(result_list) > 0:
        result_df = pd.concat(result_list, ignore_index=True)
        return result_df
    return pd.DataFrame()



def good_param_second_select(data, good_file_path='../temp/back/good_param_all_all_selected_samples_20240102_20240425.csv_param_result_list_day2_2023filter.json.csv.csv'):
    """
    最终通过好的二次参数选股
    :param data:
    :return:
    """
    result_list = []
    day_1_day_ratio_list = [0.0, 0.1]
    day_2_day_ratio_list = [0.0, 0.1]
    for day_1_day_ratio in day_1_day_ratio_list:
        for day_2_day_ratio in day_2_day_ratio_list:
            if day_1_day_ratio < day_2_day_ratio:
                continue
            good_param_df = filter_good_param(day_2_day_ratio=day_2_day_ratio, day_1_day_ratio=day_1_day_ratio, file_path=good_file_path)
            print(f"共有 day_1_day_ratio{day_1_day_ratio} day_2_day_ratio{day_2_day_ratio} {good_param_df.shape[0]} 个参数")

            tasks = [
                (row['ratio'], row['thread_day'], row['min_day_count'], row['min_select_count'], row['json_file'], row['strategy'])
                for index, row in good_param_df.iterrows()
            ]
            # tasks = tasks[:100]
            # 设定每个批次的大小
            chunk_size = 100  # 调整这个值以适应您的具体需求

            # 分批处理任务
            tasks_batches = list(chunk_tasks(tasks, chunk_size))

            with Pool(20) as pool:
                # 处理各个批次
                batch_results = pool.starmap(process_param_batch, [(data, batch) for batch in tasks_batches])

            # 扁平化结果列表
            results = [item for sublist in batch_results for item in sublist if item is not None]
            result_df = None
            # 合并所有有效结果
            if results:
                result_df = pd.concat(results, ignore_index=True)
                # 输出相同date和code对应的数量
                result_df = result_df.groupby(['date', 'code']).agg(select_count=('code', 'count'), profit_1=('profit_1', 'first'), profit_2=('profit_2', 'first'), current_price=('current_price', 'first'))
                result_df = result_df.reset_index()
                print(f"共有 {result_df.shape[0]} 个选股")
                print(result_df)
                # 将result_df写入文件
                output_filename = '../temp/back/good_param_select.csv'
                result_df['day_1_day_ratio'] = day_1_day_ratio
                result_df['day_2_day_ratio'] = day_2_day_ratio
                result_df.to_csv(output_filename, index=False)
                result_list.append(result_df)
    if len(result_list) > 0:
        result_df = pd.concat(result_list, ignore_index=True)
        return result_df
    return pd.DataFrame()

def test():
    param_file_path = '../final_zuhe/other/result_list_day1_2023filter.json'
    with open(param_file_path, 'r') as file:
        result_list = json.load(file)
    # 筛选出bad_count/total_days小于0.1的数据
    result_list = [item for item in result_list if item['bad_count'] / item['select_day_count'] <= 0.1]
    # 过滤掉select_day_count为小于1的数据
    result_list = [item for item in result_list if item['select_day_count'] >= 1]
    # 统计result_list中的相同thread_day和json_file对应的数量
    result_df = pd.DataFrame(result_list)
    result_df = result_df.groupby(['thread_day', 'json_file']).agg(select_day_count=('select_day_count', 'count'))
    result_df = result_df.reset_index()
    print(result_df)

def mul_select(all_data_frame_path):
    """
    对已经通过模型选择的数据，进行第一层参数的选择，然后再进行第二层参数的选择
    :param all_data_frame_path:
    :return:
    """
    good_file_path_dict = {'../temp/back/good_param_all_all_selected_samples_20240102_20240425.csv_param_result_list_day2_2023filter.json.csv.csv':'../final_zuhe/other/result_list_day2_2023filter.json',
                           '../temp/back/good_param_all_all_selected_samples_20240102_20240425.csv_param_result_list_day1_2023filter.json.csv.csv':'../final_zuhe/other/result_list_day1_2023filter.json',
                           '../temp/back/good_param_all_all_selected_samples_20240102_20240425.csv_param_result_list_day2.json.csv.csv':'../final_zuhe/other/result_list_day2.json',
                           '../temp/back/good_param_all_all_selected_samples_20240102_20240425.csv_param_result_list_day1.json.csv.csv':'../final_zuhe/other/result_list_day1.json'
                           }
    result_list = []
    for good_file_path, param_file_path in good_file_path_dict.items():
        print(f"开始处理 {good_file_path}")
        output_filename = get_all_param_select(all_data_frame_path, param_file_path=param_file_path)
        all_selected_samples = pd.read_csv(output_filename, dtype={'代码': str})
        all_selected_samples['ratio'] = all_selected_samples['bad_count'] / all_selected_samples['select_day_count']
        first_select_result = good_param_first_select(all_selected_samples)
        second_select_result = good_param_second_select(all_selected_samples, good_file_path=good_file_path)
        # 将first_select_result和second_select_result合并
        result_df = pd.concat([first_select_result, second_select_result], ignore_index=True)
        result_df['good_file_path'] = good_file_path
        result_list.append(result_df)
    if len(result_list) > 0:
        result_df = pd.concat(result_list, ignore_index=True)
        result_df['current_date'] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #获取当前时间，保留到天，转换为字符串
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        output_filename = f'../temp/back/good_param_select_{current_date_str}.csv'
        # 将result_df增量写入文件
        result_df.to_csv(output_filename, index=False, mode='a', header=True)

def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        traceback.print_exc()
        return {}
def get_all_model_list():
    all_model_name_dict = {}
    for root, dirs, files in os.walk('../final_zuhe/other/'):
        for file in files:
            if file.startswith('new'):
                full_name = os.path.join(root, file)
                model_info = read_json(full_name)
                all_model_name_dict[file] = model_info
    # 按照model_info的长度升序排列
    all_model_name_dict = dict(sorted(all_model_name_dict.items(), key=lambda item: len(item[1])))
    return all_model_name_dict

def gen_full_select():
    """
    生成全量的选择数据
    :return:
    """
    # 读取../model/other/下所有以new开头的json文件，并且汇总去重
    # all_data_list = []
    # for root, dirs, files in os.walk('../final_zuhe/other/'):
    #     for file in files:
    #         if file.startswith('new'):
    #             full_name = os.path.join(root, file)
    #             data = read_json(full_name)
    #             all_data_list.extend(data)
    # # 去重，以model_name为唯一标识
    # all_data = {item['model_name']: item for item in all_data_list}
    # # 将all_data变成列表
    # model_info_list = list(all_data.values())
    #
    # # 将all_data写入文件
    # with open('../final_zuhe/other/all_data.json', 'w') as file:
    #     json.dump(model_info_list, file)
    # balance_disk()

    with open('../final_zuhe/other/all_data.json', 'r') as file:
        model_info_list = json.load(file)
    file_path_list = [
        # '../train_data/profit_1_day_1_ratio_0.25/',
        # '../train_data/profit_1_day_1_ratio_0.3/',
        # '../train_data/profit_1_day_1_ratio_0.4/',
        '../train_data/profit_1_day_1_ratio_0.5/',
        # '../train_data/profit_1_day_2_ratio_0.25/',
        # '../train_data/profit_1_day_2_ratio_0.3/',
        # '../train_data/profit_1_day_2_ratio_0.4/',
        '../train_data/profit_1_day_2_ratio_0.5/',
        # '../train_data/profit_1_day_2_ratio_0.6/',
    ]
    for file_path in file_path_list:
        output_filename = f'{file_path}bad_1_select.csv'
        if not os.path.exists(output_filename):
            data_file_path = f'{file_path}bad_1.csv'
            data = low_memory_load(data_file_path)
            data['日期'] = pd.to_datetime(data['日期'])
            # 获取data_file_path的磁盘大小
            disk_size = os.path.getsize(data_file_path) / 1024 / 1024
            print(f"{data_file_path} 磁盘大小为 {disk_size} MB")
            if disk_size > 1600:
                process_count = 1
            elif disk_size > 700:
                process_count = 2
            elif disk_size > 400:
                process_count = 3
            elif disk_size > 100:
                process_count = 4
            else:
                process_count = 5
            all_selected_samples = get_all_good_data_with_model_name_list_new(data, model_info_list, process_count=process_count, thread_count=1, output_file_path=output_filename)
        output_filename = f'{file_path}good_1_select.csv'
        if not os.path.exists(output_filename):
            data_file_path = f'{file_path}good_1_merged.csv'
            data = low_memory_load(data_file_path)
            data['日期'] = pd.to_datetime(data['日期'])
            # 获取data_file_path的磁盘大小
            disk_size = os.path.getsize(data_file_path) / 1024 / 1024
            print(f"{data_file_path} 磁盘大小为 {disk_size} MB")
            if disk_size > 1600:
                process_count = 1
            elif disk_size > 700:
                process_count = 2
            elif disk_size > 400:
                process_count = 3
            elif disk_size > 100:
                process_count = 4
            else:
                process_count = 5

            all_selected_samples = get_all_good_data_with_model_name_list_new(data, model_info_list, process_count=process_count, thread_count=1, output_file_path=output_filename)

def select_first_code_process(all_model_name_dict, json_file, good_params_df, origin_selected_samples, all_result_list):
    model_info_list = all_model_name_dict[json_file]
    filter_good_params_df = good_params_df[good_params_df['json_file'] == json_file]
    # 遍历filter_good_params_df
    for index, row in filter_good_params_df.iterrows():
        date_count = row['date_count']
        is_date_count_more = row['is_date_count_more']
        cha_zhi = row['cha_zhi']
        is_cha_zhi_more = row['is_cha_zhi_more']
        min_count = row['min_count']
        strategy = row['strategy']
        is_more_min_count = row['is_more_min_count']
        if is_date_count_more:
            model_name_list = [model_info['model_name'] for model_info in model_info_list if
                               model_info['date_count'] >= date_count]
        else:
            model_name_list = [model_info['model_name'] for model_info in model_info_list if
                               model_info['date_count'] < date_count]
        date_count_origin_selected_samples = origin_selected_samples[
            origin_selected_samples['model_name'].isin(model_name_list)]
        if cha_zhi is not None:
            if is_cha_zhi_more:
                cha_zhi_origin_selected_samples = date_count_origin_selected_samples[
                    date_count_origin_selected_samples['cha_thread'] >= cha_zhi]
            else:
                cha_zhi_origin_selected_samples = date_count_origin_selected_samples[
                    date_count_origin_selected_samples['cha_thread'] < cha_zhi]
        else:
            if is_cha_zhi_more:
                cha_zhi_origin_selected_samples = date_count_origin_selected_samples.loc[
                    date_count_origin_selected_samples.groupby('日期')['cha_thread'].idxmax()]
            else:
                cha_zhi_origin_selected_samples = date_count_origin_selected_samples.loc[
                    date_count_origin_selected_samples.groupby('日期')['cha_thread'].idxmin()]
        cha_zhi_origin_selected_samples_group = cha_zhi_origin_selected_samples.groupby(['日期', '代码']).agg(
            rf_select_count=('代码', 'count'),
            min_close=('收盘', 'first'),
            后续1日最高价利润率=('后续1日最高价利润率', 'mean'),
            后续2日最高价利润率=('后续2日最高价利润率', 'mean'))
        cha_zhi_origin_selected_samples_group = cha_zhi_origin_selected_samples_group.reset_index()
        if is_more_min_count:
            min_count_origin_selected_samples_group = cha_zhi_origin_selected_samples_group[
                cha_zhi_origin_selected_samples_group['rf_select_count'] >= min_count]
        else:
            min_count_origin_selected_samples_group = cha_zhi_origin_selected_samples_group[
                cha_zhi_origin_selected_samples_group['rf_select_count'] < min_count]
        result_df = get_detail_analysis(min_count_origin_selected_samples_group)
        # 获取result_df中strategy列为strategy的数据
        result_df = result_df[result_df['strategy'] == strategy]
        # 获取result_df中total_count_date_list的值
        total_count_date_list = result_df['total_count_date_list'].values[0]
        if total_count_date_list:
            total_count_date_list = total_count_date_list.split(',')
            all_result_list.extend(total_count_date_list)
    return all_result_list

def get_first_good_param():
    # good_params_df = pd.read_csv('../final_zuhe/other/first_param_profit_1_day_1_ratio_0.25_good_1_select.csv_new_good_all_model_reports_cuml_profit_1_day_2_ratio_0.25_all.json.csv')
    # good_params_df = good_params_df[good_params_df['day_1_day'] < 1]
    # good_params_df_other = pd.read_csv('../final_zuhe/other/first_param_profit_1_day_1_ratio_0.25_bad_1_select.csv_new_good_all_model_reports_cuml_profit_1_day_2_ratio_0.25_all.json.csv')
    # good_params_df_other = good_params_df_other[good_params_df_other['day_1_day'] < 1]
    # # 帮我找到good_params_df和good_params_df_other中json_file，date_count，is_date_count_more，cha_zhi，is_cha_zhi_more，min_count，strategy，is_more_min_count这几个字段相同的数据
    # merge_columns = ['json_file', 'date_count', 'is_date_count_more', 'cha_zhi', 'is_cha_zhi_more', 'min_count',
    #                  'strategy', 'is_more_min_count']
    #
    # # 选择需要的列
    # good_params_df_subset = good_params_df[merge_columns]
    # good_params_df_other_subset = good_params_df_other[merge_columns]
    #
    # # 帮我找到good_params_df和good_params_df_other中json_file，date_count，is_date_count_more，cha_zhi，is_cha_zhi_more，min_count，strategy，is_more_min_count这几个字段相同的数据
    # same_good_params_df = pd.merge(good_params_df_subset, good_params_df_other_subset,
    #                                on=merge_columns,
    #                                how='inner')
    good_params_df = pd.read_csv('../final_zuhe/other/good_df.csv')
    # good_params_df = remove_duplicate_rows(good_params_df)
    # good_params_df = good_params_df[good_params_df['day_2_day'] <= 1]
    good_params_df = good_params_df[good_params_df['day_1_day'] <= 0]
    good_params_df = good_params_df[good_params_df['total_days'] >= 7]
    # good_params_df = good_params_df[good_params_df['total_days'] >= 5]
    # 帮我找到good_params_df和good_params_df_other中json_file，date_count，is_date_count_more，cha_zhi，is_cha_zhi_more，min_count，strategy，is_more_min_count这几个字段有重复的数据
    # good_params_df = good_params_df.groupby(['json_file', 'date_count', 'is_date_count_more', 'cha_zhi', 'is_cha_zhi_more', 'min_count', 'strategy', 'is_more_min_count']).agg(param_count=('date_count', 'count')).reset_index()
    # good_params_df = good_params_df[good_params_df['param_count'] > 1]
    return good_params_df

    # 读取'../final_zuhe/other下面的所有.csv文件
    # all_file_list = []
    # for root, dirs, files in os.walk('../final_zuhe/other/'):
    #     for file in files:
    #         if file.endswith('.csv') and 'first_param' in file:
    #             full_name = os.path.join(root, file)
    #             all_file_list.append(full_name)
    # good_df_list = []
    # for file_path in all_file_list:
    #     temp_df = pd.read_csv(file_path, dtype={'代码': str})
    #     good_df_list.append(temp_df)
    # good_df = pd.concat(good_df_list, ignore_index=True)
    # good_df = remove_duplicate_rows(good_df)
    # # 将good_df保存到文件
    # good_df.to_csv('../final_zuhe/other/good_df.csv', index=False)
    return good_df

def select_first_code(file_path='../train_data/profit_1_day_1_ratio_0.25/bad_1_select.csv'):
    all_result_list = []
    base_name = os.path.basename(file_path)
    origin_selected_samples = low_memory_load(file_path)
    compare_origin_selected_samples = low_memory_load('../train_data/2024_data_2024.csv')
    compare_origin_selected_samples['日期'] = pd.to_datetime(compare_origin_selected_samples['日期'])
    compare_origin_selected_samples = compare_origin_selected_samples[compare_origin_selected_samples.columns.drop(list(compare_origin_selected_samples.filter(regex='信号')))]
    origin_selected_samples['日期'] = pd.to_datetime(origin_selected_samples['日期'])
    if '后续1日最高价利润率' not in origin_selected_samples.columns:
        origin_selected_samples['后续1日最高价利润率'] = 0
    if '后续2日最高价利润率' not in origin_selected_samples.columns:
        origin_selected_samples['后续2日最高价利润率'] = 0
    good_params_df = get_first_good_param()
    print(f"共有 {good_params_df.shape[0]} 个参数")
    json_file_list = list(good_params_df['json_file'].unique())
    all_model_name_dict = get_all_model_list()

    # 创建进程池
    pool = Pool(processes=10)
    # 使用进程池的map方法并行处理json_file_list
    result_lists = pool.starmap(select_first_code_process, [(all_model_name_dict, json_file, good_params_df, origin_selected_samples, []) for json_file in json_file_list])
    # 关闭进程池
    pool.close()
    pool.join()

    # 合并所有进程的结果
    for result_list in result_lists:
        all_result_list.extend(result_list)

    all_result_dict = Counter(all_result_list)
    all_result_dict = dict(sorted(all_result_dict.items(), key=lambda item: item[1], reverse=True))
    print(f"{len(all_result_dict)} 个数据")
    with open(f'../final_zuhe/select/first_all_selected_{base_name}.json', 'w') as file:
        json.dump(all_result_dict, file)
    result_list = []
    for date, count in all_result_dict.items():
        date, code = date.split('_')
        if compare_origin_selected_samples[(compare_origin_selected_samples['日期'] == date) & (compare_origin_selected_samples['代码'] == code)].empty:
            print(f"{date} {code} 不存在")
        else:
            select_data = compare_origin_selected_samples[(compare_origin_selected_samples['日期'] == date) & (compare_origin_selected_samples['代码'] == code)].iloc[0]
            select_data['rf_select_count'] = count
            result_list.append(select_data)
    if len(result_list) > 0:
        result_df = pd.DataFrame(result_list)
        result_df.to_csv(f'../final_zuhe/select/first_all_selected_{base_name}.csv', index=False)




def example():
    """
    示例函数
    :return:
    """
    file_path = '../train_data/profit_1_day_1_ratio_0.5/bad_1_select.csv'
    save_all_selected_samples(file_path)
    file_path = '../train_data/profit_1_day_1_ratio_0.5/good_1_select.csv'
    save_all_selected_samples(file_path, processes=25)
    # gen_full_select()
    return

    # select_first_code('../temp/data/second_all_selected_samples_20240507_20240507.csv')


    # f'../temp/data/all_selected_samples_20240430_20240430.csv'
    # data = low_memory_load('../temp/data/all_selected_samples_20240102_20240430.csv')
    # data['日期'] = pd.to_datetime(data['日期'])
    # data = data[data['cha_thread'] >= 0]
    # results_file_path = '../final_zuhe/other/2024_all_data_performance.json'
    # performance_results = read_json(results_file_path)
    # key_name = f'后续{1}日{1}成功率'
    # ratio = 1
    # bad_ratio_day = [pd.to_datetime(date).date() for date, result in performance_results.items() if
    #                  result[key_name] <= ratio]
    # data = data[data['日期'].dt.date.isin(bad_ratio_day)]
    # # data = data[data['日期'] == '2024-04-18']
    # # 按照日期和code分组,并计算每组的行数和price的和
    # # 按照日期和代码分组,并计算每组的后续1日最高价利润率的第一个值和行数
    # group_data = data.groupby(['日期', '代码']).agg({'后续1日最高价利润率': 'first', '后续2日最高价利润率': 'first'}).reset_index()
    # group_data['count'] = data.groupby(['日期', '代码']).size().reset_index(name='count')['count']
    # # 选出每个日期数量最多的数据
    # group_data = group_data.loc[group_data.groupby('日期')['count'].idxmax()]
    # # 选出后续1日最高价利润率大于0.01的数据


    # # 使用模型在阈值范围内选股
    # with open('../final_zuhe/other/all_data.json', 'r') as file:
    #     model_info_list = json.load(file)
    # # data = low_memory_load('../final_zuhe/real_time/select_RF_2024-04-29_real_time.csv')
    # # data['日期'] = pd.to_datetime(data['日期'])
    # data = low_memory_load('../train_data/profit_1_day_1_ratio_0.7/good_1_merged.csv')
    # data['日期'] = pd.to_datetime(data['日期'])
    # all_selected_samples = get_all_good_data_with_model_name_list_new(data, model_info_list, process_count=3, thread_count=3)
    # # 对已经通过模型选择的数据，进行第一层参数的选择，然后再进行第二层参数的选择
    # data = pd.read_csv('../temp/back/good_param_select_2024-04-30.csv')
    # mul_select('../temp/data/all_selected_samples_20240430_20240430.csv')

    # # 获取第一层参数的性能
    # save_all_selected_samples(all_selected_samples)

    # # 获取第二次参数的性能
    # all_selected_samples_with_param_file = get_all_param_select('../temp/data/all_selected_samples_20240102_20240425.csv', param_file_path='../final_zuhe/other/result_list_day2.json')
    # get_good_param_by_param_select(all_selected_samples_with_param_file)



if __name__ == '__main__':
    example()
    # filter_good_param()
    # sort_choose_data_result()
    # analyse_all_select(file_path)
    # sort_all_select()
    # balance_disk()
    # analysis_model()
    # get_thread()

    # delete_bad_model()
    # print('删除完成')
    # compress_model()
    # test_load_memory_ratio()
    # pre_load_model()

    # balance_disk()
    # output_filename = '../final_zuhe/other/good_all_model_reports_cuml.json'
    # with open(output_filename, 'r') as file:
    #     sorted_scores_list = json.load(file)
    #
    # all_output_filename = '../final_zuhe/other/all_reports_cuml.json'
    # with open(output_filename, 'r') as file:
    #     all_sorted_scores_list = json.load(file)
    #
    # model_name_list = [item['model_name'] for item in sorted_scores_list]
    # all_model_name_list = [item['model_name'] for item in all_sorted_scores_list]
    #
    # # 找出all_sorted_scores_list比sorted_scores_list多出来的模型
    # bad_model_list = list(set(all_model_name_list) - set(model_name_list))

    # write_joblib_files_to_txt('/mnt/d/model/all_models/')
    # write_joblib_files_to_txt('/mnt/g/model/all_models/')
    # write_joblib_files_to_txt('../model/all_models/profit_1_day_1_bad_0.4')

    # origin_data_path = '../temp/real_time_price.csv'
    # data = pd.read_csv(origin_data_path, low_memory=False, dtype={'代码': str})
    # get_all_good_data(data)
    # data = pd.read_csv('../temp/all_selected_samples_0.csv', low_memory=False, dtype={'代码': str})
    # all_rf_model_list = load_rf_model_new(1, True)
    # 将all_rf_model_list按照score升序排序
    # all_rf_model_list = sorted(all_rf_model_list, key=lambda x: x['precision'])
    # data = pd.read_csv('../temp/all_selected_samples_50.csv', low_memory=False, dtype={'代码': str})
    # data = pd.read_csv('../temp/data/.csv', low_memory=False, dtype={'代码': str})
    # data = low_memory_load('../train_data/2023_data_all.csv')
    # data = low_memory_load('../final_zuhe/real_time/select_RF_2024-04-29_real_time.csv')
    # data['日期'] = pd.to_datetime(data['日期'])
    # data = data[data['日期'] >= '2024-01-01']
    # start = time.time()
    # with open('../final_zuhe/other/good_all_model_reports_cuml.json', 'r') as file:
    #     model_info_list = json.load(file)
    # with open('../final_zuhe/other/good_all_model_reports_cuml_new.json', 'r') as file:
    #     model_info_list = json.load(file)
    # # # 筛选出model_size在0.08到0.2之间的模型
    # # model_info_list = [model_info for model_info in model_info_list if 'thread_day_1' in model_info['model_name']]
    # all_selected_samples = get_all_good_data_with_model_name_list_new(data, model_info_list, process_count=2, thread_count=2)

    # all_selected_samples = low_memory_load('../temp/data/all_selected_samples_20240102_20240425.csv')
    # all_selected_samples['日期'] = pd.to_datetime(all_selected_samples['日期'])
    # all_selected_samples = all_selected_samples[all_selected_samples['日期'] >= '2024-04-24']
    # all_selected_samples = low_memory_load('../temp/data/all_selected_samples_20240425_20240425.csv')

    # all_selected_samples = low_memory_load('../temp/data/all_selected_samples_20230103_20231229.csv')
    # # all_selected_samples = None
    # output_filename = get_all_param_select('../temp/data/all_selected_samples_20240425_20240425.csv')
    # mul_select('../temp/data/all_selected_samples_20240425_20240425.csv')
    # get_good_param_by_param_select(output_filename)
    # all_selected_samples = pd.read_csv(output_filename, dtype={'代码': str})
    # all_selected_samples['ratio'] = all_selected_samples['bad_count'] / all_selected_samples['select_day_count']
    # good_param_first_select(all_selected_samples)
    # good_param_second_select(all_selected_samples)
    # save_all_selected_samples(all_selected_samples)
    # choose_code_from_all_selected_samples(all_selected_samples)
    # choose_code_from_all_selected_samples(all_selected_samples, param_file_path='../final_zuhe/other/result_list_day2_2023filter.json')
    # D:680G 15个 W:1440G 97个 20240415-23：26
    # D:641G 78个 W:1440G 187个 20240416-00：43

    #
    # # print(f"总耗时 {time.time() - start}")
    # # del all_selected_samples
    # # gc.collect()
    # # # 获取data中包含'信号'的列
    # # signal_columns = [column for column in data.columns if '节假日' in column]
    # # signal_data = data[signal_columns]
    # # print(signal_data)
    # # data = pd.read_csv('../train_data/2024_data_all.csv', low_memory=False, dtype={'代码': str})
    # data = low_memory_load('../train_data/2024_data.csv')
    # data['日期'] = pd.to_datetime(data['日期'])
    # # get_all_good_data_with_model_name_list_new_pre(data, 50)
    #
    #
    #
    #
    # data = data[data['日期'] >= '2024-03-20']
    # # 按照日期分组
    # data_group = data.groupby(['日期'])
    # # 初始化一个临时列表用于存储五个分组
    # temp_group_list = []
    # # 初始化一个计数器
    # counter = 0
    # with open('../final_zuhe/other/good_all_model_reports_cuml.json', 'r') as file:
    #     all_model_info_list = json.load(file)
    # all_model_info_list = [model_info for model_info in all_model_info_list if 0 < model_info['model_size'] < 0.3]
    # # 迭代每个分组
    # for date, group in data_group:
    #     # 将分组添加到临时列表
    #     temp_group_list.append(group)
    #     # 每当临时列表中有五个分组或者是最后一个分组时，执行函数
    #     if len(temp_group_list) == 5 or (counter == len(data_group) - 1):
    #         # 将五个分组的数据合并为一个DataFrame
    #         batch_data = pd.concat(temp_group_list)
    #         # 对这批数据执行函数
    #         all_selected_samples = get_all_good_data_with_model_name_list_new(batch_data, all_model_info_list, process_count=4, thread_count=3, code_list='all')
    #         # 清空临时列表，为下一个批次做准备
    #         temp_group_list = []
    #         # break
    #     # 更新计数器
    #     counter += 1
    # # load_rf_model_new()