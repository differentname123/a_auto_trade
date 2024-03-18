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
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Pool
import cudf
import pandas as pd
from joblib import dump, load
import numpy as np
from StrategyExecutor.common import low_memory_load, downcast_dtypes
from StrategyExecutor.getAllRandomForestClassifierReportSort import deal_reports

D_MODEL_PATH = '/mnt/d/model/all_models/'
G_MODEL_PATH = '/mnt/g/model/all_models/'
MODEL_PATH = '/mnt/w/project/python_project/a_auto_trade/model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/reports'
MODEL_OTHER = '/mnt/w/project/python_project/a_auto_trade/model/other'
TRAIN_DATA_PATH = '/mnt/w/project/python_project/a_auto_trade/train_data'
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

def load_existing_report(report_path):
    result_dict = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                result_dict = json.load(f)
        except json.JSONDecodeError:
            result_dict = {}
    return result_dict


def load_model(abs_name, model_name):
    try:
        model = load(abs_name)
        print(f"模型 {model_name} 加载成功。")
        return model
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(abs_name):
            # os.remove(abs_name)
            print(f"模型 {abs_name} 加载失败，跳过。")
        print(f"模型 {model_name} 不存在，跳过。")
        return None


def is_report_exists(result_dict, model_name, file_path):
    if model_name in result_dict and file_path in result_dict[model_name]:
        if result_dict[model_name][file_path] != {}:
            return True
    return False


def load_data(file_path):
    start = time.time()
    data = low_memory_load(file_path)
    print(f"加载数据 {file_path} 耗时: {time.time() - start:.2f}秒")
    return data


def process_abs_threshold(data, y_pred_proba, y_test, total_samples, abs_threshold, tree_threshold, thread_ratio,
                          this_key, temp_dict_result):
    high_confidence_true = (y_pred_proba[1] > abs_threshold)
    high_confidence_false = (y_pred_proba[0] > abs_threshold)
    if np.sum(high_confidence_true) == 0 and np.sum(high_confidence_false) == 0:
        return temp_dict_result

    selected_true = high_confidence_true & y_test
    selected_data = data[high_confidence_true]
    unique_dates = selected_data['日期'].unique().to_pandas()
    precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(high_confidence_true) > 0 else 0
    predicted_true_samples = np.sum(high_confidence_true)

    selected_false = high_confidence_false & ~y_test
    selected_data_false = data[high_confidence_false]
    unique_dates_false = selected_data_false['日期'].unique().to_pandas()
    precision_false = np.sum(selected_false) / np.sum(high_confidence_false) if np.sum(high_confidence_false) > 0 else 0
    predicted_false_samples_false = np.sum(high_confidence_false)

    daily_precision = {}
    daily_precision_false = {}
    if precision >= thread_ratio:
        for date in unique_dates:
            date_mask = selected_data['日期'] == date
            date_selected_true = selected_true[high_confidence_true]
            date_true_count = np.sum(date_selected_true[date_mask])
            date_count = np.sum(date_mask)
            daily_precision[str(date)] = {
                'precision': date_true_count / date_count if date_count > 0 else 0,
                'count': int(date_count),
                'false_count': int(date_count - date_true_count),
                'true_count': int(date_true_count)
            }
    if precision_false >= thread_ratio:
        for date in unique_dates_false:
            date_mask_false = selected_data_false['日期'] == date
            date_selected_false = selected_false[high_confidence_false]
            date_false_count = np.sum(date_selected_false[date_mask_false])
            date_count_false = np.sum(date_mask_false)
            daily_precision_false[str(date)] = {
                'precision': date_false_count / date_count_false if date_count_false > 0 else 0,
                'count': int(date_count_false),
                'false_count': int(date_false_count),
                'true_count': int(date_count_false - date_false_count)
            }

    temp_dict = create_temp_dict(tree_threshold, 0, abs_threshold, unique_dates, precision, predicted_true_samples,
                                 total_samples, unique_dates_false, precision_false, predicted_false_samples_false,
                                 thread_ratio, daily_precision, daily_precision_false)
    if this_key not in temp_dict_result:
        temp_dict_result[this_key] = []
    temp_dict_result[this_key].append(temp_dict)
    return temp_dict_result


def update_this_key_map(temp_dict_result, this_key_map, file_path, model_name):
    for key, value in temp_dict_result.items():
        temp_dict_list = sorted(value, key=lambda x: x['false_score'], reverse=True)
        false_flag = True
        true_flag = True
        if temp_dict_list[0]['false_score'] > 0:
            false_flag = False
        temp_dict_list = sorted(value, key=lambda x: x['score'], reverse=True)
        if temp_dict_list[0]['score'] > 0:
            true_flag = False
        temp_dict_result[key] = temp_dict_list
        if false_flag and true_flag:
            print(f"分数全为0 {key} 对于文件 {file_path} 模型 {model_name}的，跳过后续")
            this_key_map[key] = False


def save_report(report_path, result_dict):
    with open(report_path, 'w') as f:
        json.dump(result_dict, f)


def create_temp_dict(tree_threshold, cha_zhi_threshold, abs_threshold, unique_dates, precision, predicted_true_samples,
                     total_samples, unique_dates_false, precision_false, predicted_false_samples_false, thread_ratio,
                     daily_precision, daily_precision_false):
    temp_dict = {}
    temp_dict['tree_threshold'] = float(tree_threshold)
    temp_dict['cha_zhi_threshold'] = float(cha_zhi_threshold)
    temp_dict['abs_threshold'] = float(abs_threshold)
    temp_dict['unique_dates'] = len(unique_dates.tolist())
    temp_dict['precision'] = precision
    temp_dict['predicted_true_samples'] = int(predicted_true_samples)
    temp_dict['total_samples'] = int(total_samples)
    temp_dict['predicted_ratio'] = predicted_true_samples / total_samples if total_samples > 0 else 0
    temp_dict['unique_dates_false'] = len(unique_dates_false.tolist())
    temp_dict['precision_false'] = precision_false
    temp_dict['predicted_false_samples_false'] = int(predicted_false_samples_false)
    temp_dict['predicted_ratio_false'] = predicted_false_samples_false / total_samples if total_samples > 0 else 0
    temp_dict['score'] = precision * temp_dict['predicted_ratio'] * 100 if precision > thread_ratio else 0
    temp_dict['false_score'] = precision_false * temp_dict[
        'predicted_ratio_false'] * 100 if precision_false > thread_ratio else 0
    temp_dict['daily_precision'] = daily_precision
    temp_dict['daily_precision_false'] = daily_precision_false
    return temp_dict


def process_cha_zhi_threshold(data, y_pred_proba, y_test, total_samples, cha_zhi_threshold, tree_threshold,
                              thread_ratio, this_key, temp_dict_result):
    proba_diff = y_pred_proba[1] - y_pred_proba[0]
    high_confidence_diff = (proba_diff > cha_zhi_threshold)
    proba_diff_neg = y_pred_proba[0] - y_pred_proba[1]
    high_confidence_diff_neg = (proba_diff_neg > cha_zhi_threshold)
    if np.sum(high_confidence_diff) == 0 and np.sum(high_confidence_diff_neg) == 0:
        return temp_dict_result

    selected_data_diff = data[high_confidence_diff]
    unique_dates_diff = selected_data_diff['日期'].unique().to_pandas()
    selected_true_diff = high_confidence_diff & y_test
    precision_diff = np.sum(selected_true_diff) / np.sum(high_confidence_diff) if np.sum(
        high_confidence_diff) > 0 else 0
    predicted_true_samples_diff = np.sum(high_confidence_diff)

    selected_data_diff_neg = data[high_confidence_diff_neg]
    unique_dates_diff_neg = selected_data_diff_neg['日期'].unique().to_pandas()
    selected_false_diff_neg = high_confidence_diff_neg & ~y_test
    precision_false = np.sum(selected_false_diff_neg) / np.sum(high_confidence_diff_neg) if np.sum(
        high_confidence_diff_neg) > 0 else 0
    predicted_false_samples_diff_neg = np.sum(high_confidence_diff_neg)

    daily_precision_diff = {}
    daily_precision_diff_neg = {}
    if precision_diff >= thread_ratio or precision_false >= thread_ratio:
        for date in unique_dates_diff:
            date_mask = selected_data_diff['日期'] == date
            date_selected_true_diff = selected_true_diff[high_confidence_diff]
            date_true_count = np.sum(date_selected_true_diff[date_mask])
            date_count = np.sum(date_mask)
            daily_precision_diff[str(date)] = {
                'precision': date_true_count / date_count if date_count > 0 else 0,
                'count': int(date_count),
                'false_count': int(date_count - date_true_count),
                'true_count': int(date_true_count)
            }
        for date in unique_dates_diff_neg:
            date_mask_neg = selected_data_diff_neg['日期'] == date
            date_selected_false_diff_neg = selected_false_diff_neg[high_confidence_diff_neg]
            date_false_count = np.sum(date_selected_false_diff_neg[date_mask_neg])
            date_count_neg = np.sum(date_mask_neg)
            daily_precision_diff_neg[str(date)] = {
                'precision': date_false_count / date_count_neg if date_count_neg > 0 else 0,
                'count': int(date_count_neg),
                'false_count': int(date_false_count),
                'true_count': int(date_count_neg - date_false_count)
            }

    temp_dict = create_temp_dict(tree_threshold, cha_zhi_threshold, 0, unique_dates_diff, precision_diff,
                                 predicted_true_samples_diff, total_samples, unique_dates_diff_neg, precision_false,
                                 predicted_false_samples_diff_neg, thread_ratio, daily_precision_diff,
                                 daily_precision_diff_neg)
    if this_key not in temp_dict_result:
        temp_dict_result[this_key] = []
    temp_dict_result[this_key].append(temp_dict)
    return temp_dict_result

def process_pred_proba(data, y_pred_proba, y_test, total_samples, abs_threshold_values, cha_zhi_values, thread_ratio,
                       this_key_map, temp_dict_result):
    start_time = time.time()
    for abs_threshold in abs_threshold_values:
        this_key = 'tree_0_abs_1'
        if not this_key_map[this_key]:
            break

        temp_dict_result = process_abs_threshold(data, y_pred_proba, y_test, total_samples, abs_threshold, 0,
                                                 thread_ratio, this_key, temp_dict_result)
    print(f"abs 耗时 {time.time() - start_time:.2f}秒 模型 abs 的报告已生成 平均耗时: {(time.time() - start_time)/ len(abs_threshold_values):.2f}秒")
    return temp_dict_result


def get_model_report(abs_name, model_name, file_path, data, X_test):
    """
    为单个模型生成报告，并更新模型报告文件
    :param model_path: 模型路径
    :param model_name: 当前正在处理的模型名称
    """
    try:
        start_time = time.time()
        report_path = os.path.join(MODEL_REPORT_PATH, model_name + '_report.json')
        result_dict = load_existing_report(report_path)
        thread_ratio = 0.9
        new_temp_dict = {}
        if is_report_exists(result_dict, model_name, file_path):
            new_temp_dict[file_path] = result_dict[model_name][file_path]
            print(f"模型 {model_name} 对于文件 {file_path} 的报告已存在，跳过。")
            return

        model = load_model(abs_name, model_name)
        if model is None:
            return
        abs_threshold_values = np.arange(0.5, 1, 0.01)
        cha_zhi_values = np.arange(0.01, 1, 0.05)
        this_key_map = {
            'tree_1_abs_1': False, 'tree_0_abs_1': True,
            'tree_1_cha_zhi_1': False, 'tree_0_cha_zhi_1': True
        }
        file_start_time = time.time()
        temp_dict_result = {}
        print("加载数据{}...".format(file_path))
        profit = int(model_name.split('profit_')[1].split('_')[0])
        thread_day = int(model_name.split('thread_day_')[1].split('_')[0])
        key_name = f'后续{thread_day}日最高价利润率'
        y_test = data[key_name] >= profit
        total_samples = len(y_test)
        print(f"处理数据耗时: {time.time() - file_start_time:.2f}秒")


        if this_key_map['tree_0_abs_1'] or this_key_map["tree_0_cha_zhi_1"]:
            start = time.time()
            y_pred_proba = model.predict_proba(X_test)
            print(f" 预测耗时: {time.time() - start:.2f}秒 {model_name}")
            temp_dict_result = process_pred_proba(data, y_pred_proba, y_test, total_samples, abs_threshold_values,
                                                  cha_zhi_values, thread_ratio, this_key_map, temp_dict_result)
        print(f"生成报告耗时: {time.time() - file_start_time:.2f}秒 获取模型 {model_name} 对于文件 {file_path} 的报告")
        new_temp_dict[file_path] = temp_dict_result
        result_dict[model_name] = new_temp_dict
        save_report(report_path, result_dict)
        end_time = time.time()
        print(f"整体耗时 {end_time - start_time:.2f} 秒模型报告已生成: {model_name}，\n\n")

        return result_dict
    except Exception as e:
        traceback.print_exc()
        print(f"生成报告时出现异常: {e}")
        return {}


def get_all_model_report():
    """
    使用多进程获取所有模型的报告。
    """
    while True:
        # 获取MODEL_REPORT_PATH下所有模型的报告
        report_list = []
        for root, ds, fs in os.walk(MODEL_REPORT_PATH):
            for f in fs:
                if f.endswith('report.json'):
                    report_list.append(f.split('_report.json')[0])
        model_list = []
        for model_path in MODEL_PATH_LIST:
        # 获取所有模型的文件名
            for root, ds, fs in os.walk(model_path):
                for f in fs:
                    full_name = os.path.join(root, f)
                    if f.endswith('joblib') and f not in report_list:
                        model_list.append((full_name, f))
        # 随机打乱model_list
        # random.shuffle(model_list)
        start_time = time.time()
        file_path = '/mnt/w/project/python_project/a_auto_trade/train_data/all_data.csv'
        print(f"开始处理数据集: {file_path}")
        data = cudf.read_csv(file_path)
        # data = low_memory_load(file_path)
        # data = cudf.DataFrame(data)
        memory = data.memory_usage(deep=True).sum()
        print(f"原始数据集内存: {memory / 1024 ** 2:.2f} MB")
        data = downcast_dtypes(data)
        memory = data.memory_usage(deep=True).sum()
        print(f"转换后数据集内存: {memory / 1024 ** 2:.2f} MB")
        signal_columns = [column for column in data.columns if '信号' in column]
        X_test = data[signal_columns]
        data = data.drop(signal_columns, axis=1)
        print(f"待训练的模型数量: {len(model_list)} 耗时: {time.time() - start_time:.2f}秒")
        for full_name, model_name in model_list:
            get_model_report(full_name, model_name, file_path, data, X_test)


if __name__ == '__main__':
    get_all_model_report()
