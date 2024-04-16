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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第一张GPU（2080），索引从0开始
import json
import time
import traceback
import cudf
from joblib import dump, load
import numpy as np
from StrategyExecutor.common import low_memory_load, downcast_dtypes
import gc
D_MODEL_PATH = '/mnt/d/model/all_models/'
G_MODEL_PATH = '/mnt/g/model/all_models/'
MODEL_PATH = '/mnt/w/project/python_project/a_auto_trade/model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/reports'
DELETED_MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/deleted_reports'
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

def process_abs_threshold(data, y_pred_proba, y_test, total_samples, abs_threshold, tree_threshold, thread_ratio,
                          this_key, temp_dict_result):
    high_confidence_true = (y_pred_proba[1] > abs_threshold)
    high_confidence_false = (y_pred_proba[0] > abs_threshold)
    if np.sum(high_confidence_true) == 0:
        print(f"abs_threshold {abs_threshold} 无预测值")
        return temp_dict_result, False

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
    find_true_flag = False
    daily_precision = {}
    daily_precision_false = {}
    true_stocks_set = []
    false_stocks_set = []
    date_precision = 0
    date_false_precision = 0
    if precision >= thread_ratio:
        good_date_count = 0
        kui_count = 0
        for date in unique_dates:
            date_mask = selected_data['日期'] == date
            date_selected_true = selected_true[high_confidence_true]
            date_true_count = np.sum(date_selected_true[date_mask])
            date_count = np.sum(date_mask)
            date_precision = date_true_count / date_count if date_count > 0 else 0
            daily_precision[str(date)] = {
                'precision': date_precision,
                'count': int(date_count),
                'false_count': int(date_count - date_true_count),
                'true_count': int(date_true_count)
            }
            kui = 3 * daily_precision[str(date)]['false_count'] - daily_precision[str(date)]['true_count']
            if kui > 0:
                kui_count += 1
            if date_precision >= 0.9:
                good_date_count += 1
        if (good_date_count / len(unique_dates) >= 0.9) and (kui_count / len(unique_dates) <= 0.05):
            date_precision = good_date_count / len(unique_dates) if len(unique_dates) > 0 else 0
            true_stocks_set = list((selected_data['代码'].to_pandas().astype(str) + selected_data['日期'].to_pandas()))
            find_true_flag = True
        else:
            daily_precision = {}
    # if precision_false >= thread_ratio:
    #     good_date_count = 0
    #     for date in unique_dates_false:
    #         date_mask_false = selected_data_false['日期'] == date
    #         date_selected_false = selected_false[high_confidence_false]
    #         date_false_count = np.sum(date_selected_false[date_mask_false])
    #         date_count_false = np.sum(date_mask_false)
    #         date_false_precision = date_false_count / date_count_false if date_count_false > 0 else 0
    #         daily_precision_false[str(date)] = {
    #             'precision': date_false_precision,
    #             'count': int(date_count_false),
    #             'false_count': int(date_false_count),
    #             'true_count': int(date_count_false - date_false_count)
    #         }
    #         if date_false_precision >= 0.9:
    #             good_date_count += 1
    #     if good_date_count / len(unique_dates_false) >= 0.9:
    #         date_false_precision = good_date_count / len(unique_dates_false) if len(unique_dates_false) > 0 else 0
    #         false_stocks_set = list((selected_data_false['代码'].to_pandas().astype(str) + selected_data_false['日期'].to_pandas()))
    #     else:
    daily_precision_false = {}
    other_dict = {}
    other_dict['date_precision'] = date_precision
    other_dict['date_false_precision'] = date_false_precision
    temp_dict = create_temp_dict(tree_threshold, 0, abs_threshold, unique_dates, precision, predicted_true_samples,
                                 total_samples, unique_dates_false, precision_false, predicted_false_samples_false,
                                 thread_ratio, daily_precision, daily_precision_false, true_stocks_set, false_stocks_set, other_dict)
    if this_key not in temp_dict_result:
        temp_dict_result[this_key] = []
    temp_dict_result[this_key].append(temp_dict)
    return temp_dict_result, find_true_flag

def save_report(report_path, result_dict):
    with open(report_path, 'w') as f:
        json.dump(result_dict, f)

def create_temp_dict(tree_threshold, cha_zhi_threshold, abs_threshold, unique_dates, precision, predicted_true_samples,
                     total_samples, unique_dates_false, precision_false, predicted_false_samples_false, thread_ratio,
                     daily_precision, daily_precision_false, true_stocks_set, false_stocks_set, other_dict):
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
    temp_dict['true_stocks_set'] = true_stocks_set
    temp_dict['false_stocks_set'] = false_stocks_set
    for key, value in other_dict.items():
        temp_dict[key] = value
    return temp_dict

def process_pred_proba(data, y_pred_proba, y_test, total_samples, abs_threshold_values, cha_zhi_values, thread_ratio,
                       this_key_map, temp_dict_result):
    start_time = time.time()
    for abs_threshold in abs_threshold_values:
        this_key = 'tree_0_abs_1'
        if not this_key_map[this_key]:
            break

        temp_dict_result, find_true_flag = process_abs_threshold(data, y_pred_proba, y_test, total_samples, abs_threshold, 0,
                                                 thread_ratio, this_key, temp_dict_result)
        if find_true_flag:
            break
    print(f"find_true_flag {find_true_flag} abs 耗时 {time.time() - start_time:.2f}秒 模型 abs 的报告已生成 平均耗时: {(time.time() - start_time)/ len(abs_threshold_values):.2f}秒 阈值列表长度: {len(temp_dict_result[this_key])}")
    return temp_dict_result

def get_model_report(abs_name, model_name, file_path, data, X_test):
    """
    为单个模型生成报告,并更新模型报告文件
    :param model_path: 模型路径
    :param model_name: 当前正在处理的模型名称
    """
    try:
        start_time = time.time()
        report_path = os.path.join(MODEL_REPORT_PATH, model_name + '_report.json')
        result_dict = load_existing_report(report_path)
        if result_dict == {}:
            report_path = os.path.join(DELETED_MODEL_REPORT_PATH, model_name + '_report.json')
            result_dict = load_existing_report(report_path)
        report_path = os.path.join(MODEL_REPORT_PATH, model_name + '_report.json')
        thread_ratio = 0.9
        new_temp_dict = {}
        if is_report_exists(result_dict, model_name, file_path):
            new_temp_dict[file_path] = result_dict[model_name][file_path]
            print(f"模型 {model_name} 对于文件 {file_path} 的报告已存在,跳过。")
            return
        file_start_time = time.time()
        model = load_model(abs_name, model_name)
        if model is None:
            return
        abs_threshold_values = np.arange(0.5, 1, 0.01)
        cha_zhi_values = np.arange(0.01, 1, 0.05)
        this_key_map = {
            'tree_1_abs_1': False, 'tree_0_abs_1': True,
            'tree_1_cha_zhi_1': False, 'tree_0_cha_zhi_1': True
        }
        temp_dict_result = {}
        temp_dict_result['model_size'] = round(os.path.getsize(abs_name) / (1024 ** 2), 2)
        print("加载数据{}...".format(file_path))
        profit = int(model_name.split('profit_')[1].split('_')[0])
        thread_day = int(model_name.split('thread_day_')[1].split('_')[0])
        key_name = f'后续{thread_day}日最高价利润率'
        y_test = data[key_name] >= profit
        total_samples = len(y_test)
        print(f"当前时间{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} 处理数据耗时: {time.time() - file_start_time:.2f}秒 模型大小: {os.path.getsize(abs_name) / 1024 ** 2:.2f}M {model_name}条数据")

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
        print(
            f"当前时间{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}  整体耗时 {end_time - start_time:.2f} 秒模型报告已生成: {model_name},\n\n")
        del model
        del y_pred_proba
        gc.collect()
        return result_dict
    except BaseException as e:
        traceback.print_exc()
        # os.remove(abs_name)
        # print(f"已删除生成报告时出现异常: {e}")
        return {}


def load_test_data():
    file_path = '/mnt/w/project/python_project/a_auto_trade/train_data/all_data.csv'
    print(f"开始处理数据集: {file_path}")
    data = low_memory_load(file_path)
    data = cudf.DataFrame(data)

    key_signal_columns = [column for column in data.columns if '最高价利润率' in column]
    key_signal_columns.extend(['日期', '代码'])
    final_data = data[key_signal_columns]
    memory = data.memory_usage(deep=True).sum()
    print(f"原始数据集内存: {memory / 1024 ** 2:.2f} MB")
    data = downcast_dtypes(data)
    memory = data.memory_usage(deep=True).sum()
    print(f"转换后数据集内存: {memory / 1024 ** 2:.2f} MB")
    signal_columns = [column for column in data.columns if '信号' in column]
    X_test = data[signal_columns]
    return final_data, X_test


def get_all_model_report(max_size=0.5, min_size=0):
    """
    使用多进程获取所有模型的报告。
    """
    final_data, X_test = None, None
    while True:
        # 获取MODEL_REPORT_PATH下所有模型的报告
        report_list = []
        for root, ds, fs in os.walk(MODEL_REPORT_PATH):
            for f in fs:
                if f.endswith('report.json'):
                    report_list.append(f.split('_report.json')[0])
        for root, ds, fs in os.walk(DELETED_MODEL_REPORT_PATH):
            for f in fs:
                if f.endswith('report.json'):
                    report_list.append(f.split('_report.json')[0])
        model_list = []
        size_list = []

        for model_path in MODEL_PATH_LIST:
            # 获取所有模型的文件名
            for root, ds, fs in os.walk(model_path):
                for f in fs:
                    full_name = os.path.join(root, f)
                    if 'good_models' in full_name:
                        continue
                    # 获取full_name文件的大小,如果大于4G,则跳过
                    file_size = os.path.getsize(full_name)
                    if file_size > max_size * 1024 ** 2 or file_size < min_size * 1024 ** 2:
                        # print(f"模型 {full_name} 大小超过4G,跳过。")
                        continue
                    if f.endswith('joblib') and f not in report_list:
                        model_list.append((full_name, f))
                        size_list.append(file_size)

        # 根据文件大小对索引进行排序
        sorted_indices = sorted(range(len(size_list)), key=lambda i: size_list[i], reverse=True)

        # 根据排序后的索引重新生成model_list
        sorted_model_list = [model_list[i] for i in sorted_indices]

        # 将排序后的列表赋值给model_list
        model_list = sorted_model_list
        print(
            f"待训练的模型数量: {len(model_list)} 已存在的模型报告数量{len(report_list)}")

        start_time = time.time()
        if final_data is None or X_test is None:
            final_data, X_test = load_test_data()
        print(
            f"待训练的模型数量: {len(model_list)} 已存在的模型报告数量{len(report_list)} 耗时: {time.time() - start_time:.2f}秒")
        file_path = '/mnt/w/project/python_project/a_auto_trade/train_data/all_data.csv'
        for full_name, model_name in model_list:
            result_dict = get_model_report(full_name, model_name, file_path, final_data, X_test)

if __name__ == '__main__':
    get_all_model_report(300, 0)
