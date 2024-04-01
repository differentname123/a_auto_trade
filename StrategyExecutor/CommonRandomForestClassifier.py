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
import multiprocessing
import os
import time
import traceback
import warnings
from multiprocessing import Pool
from multiprocessing import Process, current_process
from itertools import chain, zip_longest
import shutil

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
            print(selected_samples['日期'].value_counts())
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
                print(selected_samples['日期'].value_counts())
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
        print(selected_samples['日期'].value_counts())
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


def balance_disk(class_key='/mnt/w'):
    """
    平衡两个磁盘中的数据大小
    :param all_rf_model_list:
    :return:
    """
    final_output_filename = '../final_zuhe/other/good_all_model_reports_cuml.json'
    with open(final_output_filename, 'r') as file:
        all_model_info_list = json.load(file)
    w_model_path = MODEL_PATH
    other_model_path = D_MODEL_PATH
    all_model_info_list_w = [model_info for model_info in all_model_info_list if class_key in model_info['model_path']]
    all_model_info_list_other = [model_info for model_info in all_model_info_list if
                                 class_key not in model_info['model_path']]

    # 计算两个列表中model_size的总和
    total_size_w = sum(model_info['model_size'] for model_info in all_model_info_list_w)
    total_size_other = sum(model_info['model_size'] for model_info in all_model_info_list_other)
    print(f"all_model_info_list_w {len(all_model_info_list_w)} all_model_info_list_other {len(all_model_info_list_other)}")
    # 确定需要移动的方向和数量
    if total_size_w > total_size_other:
        move_from = all_model_info_list_w
        move_to = all_model_info_list_other
        move_path = other_model_path
    else:
        move_from = all_model_info_list_other
        move_to = all_model_info_list_w
        move_path = w_model_path
    initial_sign = 1 if total_size_w > total_size_other else -1
    current_sign = initial_sign
    # 按照model_size从大到小排序
    move_from.sort(key=lambda x: x['model_size'], reverse=True)
    # 移动模型直到两个列表的大小尽量相当
    while initial_sign == current_sign:
        model_info = move_from.pop(0)
        src_path = model_info['model_path']
        dst_dir = os.path.join(move_path, os.path.dirname(
            os.path.relpath(src_path, start=w_model_path if move_path == other_model_path else other_model_path)))

        # 检查目标目录是否存在,不存在则创建
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        # 移动文件
        shutil.move(src_path, dst_path)

        model_info['model_path'] = dst_path
        move_to.append(model_info)

        if move_path == other_model_path:
            total_size_w -= model_info['model_size']
            total_size_other += model_info['model_size']
        else:
            total_size_w += model_info['model_size']
            total_size_other -= model_info['model_size']

        print(f"Moved {src_path} to {dst_path}")
        current_sign = 1 if total_size_w > total_size_other else -1

    print(f"大小平衡后 all_model_info_list_w {len(all_model_info_list_w)} size {total_size_w} all_model_info_list_other {len(all_model_info_list_other)} size {total_size_other}")
    all_model_info_list_w.extend(all_model_info_list_other)
    all_model_info_list = all_model_info_list_w
    with open(final_output_filename, 'w') as file:
        json.dump(all_model_info_list, file)
    return all_model_info_list

def load_rf_model_new(date_count_threshold=100, need_filter=True, need_balance=False, model_max_size=100):
    """
    加载随机森林模型
    :param model_path:
    :return:
    """
    all_rf_model_list = []
    output_filename = '../final_zuhe/other/all_model_reports_cuml.json'
    final_output_filename = '../final_zuhe/other/good_all_model_reports_cuml.json'
    model_file_list = []
    for model_path in MODEL_PATH_LIST:
        # 获取model_path下的所有文件的全路径，如果是目录还需要继续递归
        for root, dirs, files in os.walk(model_path):
            for file in files:
                model_file_list.append(os.path.join(root, file))
    exist_stocks_list = []
    exist_stocks = set()
    # 加载output_filename，找到最好的模型
    with open(output_filename, 'r') as file:
        sorted_scores_list = json.load(file)
        for sorted_scores in sorted_scores_list:
            model_name = sorted_scores['model_name']
            model_file_path = None
            if need_filter:
                current_stocks = set(sorted_scores['true_stocks_set'])
                # exist_flag = False
                # # 判断current_stocks是否被包含在exist_stocks中
                # for exist_stocks in exist_stocks_list:
                #     if len(current_stocks - exist_stocks) == 0:
                #         print(f"模型 {model_name} 已经有相似的模型，跳过。")
                #         exist_flag = True
                #         break
                # if exist_flag:
                #     continue
                # exist_stocks_list.append(current_stocks)
                if len(current_stocks - exist_stocks) == 0:
                    print(f"模型 {model_name} 已经有相似的模型，跳过。")
                    continue
                exist_stocks = exist_stocks | current_stocks
            for model_path in model_file_list:
                if model_name in model_path:
                    model_file_path = model_path
                    break
            sorted_scores['true_stocks_set'] = []
            if model_file_path is not None:
                # 判断model_file_path大小是否大于model_max_size
                model_size = round(os.path.getsize(model_file_path) / (1024 ** 3), 2)
                if model_size > model_max_size:
                    print(f"{os.path.getsize(model_file_path)}大小超过 {model_max_size}G，跳过。")
                    continue
                if sorted_scores['max_score'] > date_count_threshold:
                    other_dict = sorted_scores
                    other_dict['model_path'] = model_file_path
                    other_dict['model_size'] = model_size
                    all_rf_model_list.append(other_dict)
    print(f"加载了 {len(all_rf_model_list)} 个模型")
    # 将all_rf_model_list存入final_output_filename
    with open(final_output_filename, 'w') as file:
        json.dump(all_rf_model_list, file)
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
        print(f"整体进度 {count}/{len(all_rf_model_list)} score {score} threshold {threshold}  basic_threshold {rf_model_map['threshold']} 耗时 {time.time() - start} 模型 {model_name}\n\n")
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
            print(f"整体进度 {count}/{len(all_rf_model_list)} score {score} threshold {threshold}  basic_threshold {rf_model_map['threshold']} 耗时 {time.time() - start} 模型 {model_name}\n\n")
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
    if abs_threshold > 0:
        threshold = abs_threshold
        high_confidence_true = (y_pred_proba[1] > threshold)
        predicted_true_samples = np.sum(high_confidence_true)
        # 如果有高于阈值的预测样本，打印对应的原始数据
        if predicted_true_samples > min_day:
            # 直接使用布尔索引从原始data中选择对应行
            selected_samples = X1[high_confidence_true].copy()
            # 统计selected_samples中 收盘 的和
            close_sum = selected_samples['收盘'].sum()
            print(f'高于阈值 {threshold:.2f} 的预测样本对应的原始数据:{close_sum} 代码:{set(selected_samples["代码"].values)} 收盘最小值:{selected_samples["收盘"].min()} 收盘最大值:{selected_samples["收盘"].max()}')
            print(selected_samples['日期'].value_counts())
            return selected_samples
    return selected_samples

def get_proba_data(data, rf_classifier):
    signal_columns = [column for column in data.columns if '信号' in column]
    # 获取data中在signal_columns中的列
    X = data[signal_columns]
    # 获取data中去除signal_columns中的列
    X1 = data.drop(signal_columns, axis=1)
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
    try:
        rf_model = load(rf_model_map['model_path'])
        load_time = time.time() - start

        value = rf_model_map

        y_pred_proba, X1 = get_proba_data(data, rf_model)
        selected_samples = get_thread_data_new_tree_0(y_pred_proba, X1, min_day=value['min_day'], abs_threshold=value['abs_threshold'])
        if selected_samples is not None:
            selected_samples['param'] = str(value)
            selected_samples['model_name'] = model_name
            all_selected_samples_list.append(selected_samples)
        if len(all_selected_samples_list) > 0:
            return pd.concat(all_selected_samples_list)
    except Exception as e:
        traceback.print_exc()
        print(f"模型 {model_name} 加载失败 {e}")
        # # 删除模型文件
        # if os.path.exists(rf_model_map['model_path']):
        #     os.remove(rf_model_map['model_path'])
        #     print(f"删除模型 {model_name} 成功")
    finally:
        elapsed_time = time.time() - start
        print(f"模型 {model_name} 耗时 {elapsed_time} 加载耗时 {load_time}")

def model_worker(model_info_list, data, result_list):
    print(f"Process {current_process().name} started.")
    with ThreadPoolExecutor(max_workers=4) as executor:
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

def get_all_good_data_with_model_name_list_new(data, date_count_threshold=50):
    start = time.time()
    with open('../final_zuhe/other/good_all_model_reports_cuml.json', 'r') as file:
        all_model_info_list = json.load(file)
    # 将all_model_info_list按照model_path分类，包含‘/mnt/w’的为一类，其余为一类
    all_model_info_list_w = [model_info for model_info in all_model_info_list if '/mnt/w' in model_info['model_path']]
    all_model_info_list_other = [model_info for model_info in all_model_info_list if '/mnt/w' not in model_info['model_path']]
    all_model_info_list_w.sort(key=lambda x: x['model_size'], reverse=True)
    all_model_info_list_other.sort(key=lambda x: x['model_size'], reverse=False)
    # 将all_model_info_list_w和all_model_info_list_other交错合并为all_model_info_list
    print(f"all_model_info_list_w {len(all_model_info_list_w)} all_model_info_list_other {len(all_model_info_list_other)}")
    all_model_info_list = interleave_lists(all_model_info_list_w, all_model_info_list_other)
    print(f"总共加载了 {len(all_model_info_list)} 个模型，date_count_threshold={date_count_threshold}")
    thread_count = 5
    # 分割模型列表以分配给每个进程
    chunk_size = len(all_model_info_list) // thread_count
    model_chunks = [all_model_info_list[i:i + chunk_size] for i in range(0, len(all_model_info_list), chunk_size)]

    # 存储最终结果的列表
    manager = multiprocessing.Manager()
    result_list = manager.list()

    # 创建并启动进程
    processes = []
    for i in range(thread_count):
        p = Process(target=model_worker, args=(model_chunks[i], data, result_list))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    all_selected_samples = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    all_selected_samples.to_csv(f'../temp/all_selected_samples_{date_count_threshold}.csv', index=False)

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

if __name__ == '__main__':
    # delete_bad_model()
    # print('删除完成')
    # compress_model()

    # balance_disk()


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
    # data = pd.read_csv('../temp/all_selected_samples_0.csv', low_memory=False, dtype={'代码': str})
    data = low_memory_load('../final_zuhe/real_time/select_RF_2024-04-01_real_time.csv')
    # data = pd.read_csv('../train_data/2024_data_all.csv', low_memory=False, dtype={'代码': str})
    # data = low_memory_load('../train_data/2024_data_all.csv')
    data['日期'] = pd.to_datetime(data['日期'])
    data = data[data['日期'] >= '2024-03-01']
    # 截取data最后4000行
    # data = data.iloc[-4000:]
    # data = {}
    get_all_good_data_with_model_name_list_new(data, 50)
    # load_rf_model_new()

