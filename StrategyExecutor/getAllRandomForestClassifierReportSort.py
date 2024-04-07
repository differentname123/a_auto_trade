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
import shutil
import time
import traceback
from collections import Counter
from multiprocessing import Process, Pool

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from StrategyExecutor.CommonRandomForestClassifier import load_rf_model_new

D_MODEL_PATH = '/mnt/d/model/all_models/'
G_MODEL_PATH = '/mnt/g/model/all_models/'
MODEL_PATH = '/mnt/w/project/python_project/a_auto_trade/model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/reports'
DELETED_MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/deleted_reports'
MODEL_OTHER = '../model/other'
TRAIN_DATA_PATH = '../train_data'
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=UserWarning)

def train_and_dump_model(clf, X_train, y_train, model_file_path):
    """
    训练模型并保存到指定路径
    :param clf: 分类器实例
    :param X_train: 训练数据集特征
    :param y_train: 训练数据集标签
    :param model_path: 模型保存路径
    :param model_name: 模型名称
    """
    # model_path = 'D:\model/all_models'
    # 获取model_file_path的文件名
    model_name = os.path.basename(model_file_path)
    out_put_path = model_file_path
    # 如果out_put_path目录不存在，则创建
    if not os.path.exists(os.path.dirname(out_put_path)):
        os.makedirs(os.path.dirname(out_put_path))
    # 创建一个空的out_put_path文件
    with open(out_put_path, 'w') as f:
        pass
    print(f"开始训练模型: {model_name}")
    clf.fit(X_train, y_train)
    dump(clf, model_file_path)
    print(f"模型已保存: {model_name}")
    # get_model_report(model_path, model_name)

def train_all_model(file_path_path, profit=1,thread_day_list=None, is_skip=True):
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
    signal_columns = [column for column in data.columns if '信号' in column]
    X = data[signal_columns]
    ratio_result_path = os.path.join(MODEL_OTHER, origin_data_path_dir + 'ratio_result.json')
    try:
        # 尝试加载ratio_result
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
            true_ratio = sum(y) / len(y)
            ratio_result[ratio_key] = true_ratio
            with open(ratio_result_path, 'w') as f:
                json.dump(ratio_result, f)
        print(f"处理天数阈值: {thread_day}, 真实比率: {true_ratio:.4f}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_models(X_train, y_train, 'RandomForest', thread_day, true_ratio, is_skip, origin_data_path_dir)
        train_models(X_train, y_train, 'GradientBoosting', thread_day, true_ratio, is_skip, origin_data_path_dir)


def train_all_model_grad(file_path_path, thread_day_list=None, is_skip=True):
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
    ratio_result_path = os.path.join(MODEL_OTHER, origin_data_path_dir + 'ratio_result.json')
    try:
        # 尝试加载ratio_result
        with open(ratio_result_path, 'r') as f:
            ratio_result = json.load(f)
    except FileNotFoundError:
        ratio_result= {}
    for thread_day in thread_day_list:
        y = data['Days Held'] <= thread_day
        ratio_key = origin_data_path_dir + '_' + str(thread_day)
        if ratio_key in ratio_result:
            true_ratio = ratio_result[ratio_key]
        else:
            true_ratio = sum(y) / len(y)
            ratio_result[ratio_key] = true_ratio
            with open(ratio_result_path, 'w') as f:
                json.dump(ratio_result, f)
        print(f"处理天数阈值: {thread_day}, 真实比率: {true_ratio:.4f}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_models(X_train, y_train, 'GradientBoosting', thread_day, true_ratio, is_skip, origin_data_path_dir)
        train_models(X_train, y_train, 'RandomForest', thread_day, true_ratio, is_skip, origin_data_path_dir)


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
        model_file_path = os.path.join(MODEL_PATH, origin_data_path_dir, model_name)
        if is_skip:
            flag = False
            if os.path.exists(model_file_path):
                print(f"模型 {model_name} 已存在，跳过训练。")
                flag = True
                continue
            if flag:
                continue
        clf = RandomForestClassifier(**params) if model_type == 'RandomForest' else GradientBoostingClassifier(**params)
        train_and_dump_model(clf, X_train, y_train, model_file_path)

        # 对于类别不平衡的处理
        print("处理类别不平衡...")
        # 检查并转换布尔列为整数
        X_train = X_train.astype(int)
        # 现在使用SMOTE应该不会报错
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        model_name_smote = f"smote_{model_name}"
        train_and_dump_model(clf, X_train_smote, y_train_smote, model_file_path)

def add_score(final_result, value, is_false=False):
    max_score = 0
    for k, v in final_result.items():
        if not is_false:
            final_result[k]['score'] = 0
        else:
            final_result[k]['false_score'] = 0
        for k1, v1 in value.items():
            # print(k1)
            for detail in v1[k]:
                if v['tree_threshold'] == detail['tree_threshold'] and v['cha_zhi_threshold'] == detail['cha_zhi_threshold'] and v['abs_threshold'] == detail['abs_threshold']:
                    if not is_false:
                        final_result[k]['score'] += (detail['score'])
                        # print(detail['score'])
                    else:
                        final_result[k]['false_score'] *= (detail['false_score'])
        if not is_false:
            if final_result[k]['score'] > max_score:
                max_score = final_result[k]['score']
        else:
            if final_result[k]['false_score'] > max_score:
                max_score = final_result[k]['false_score']
    return max_score

def maintain_scores(temp_score, value, min_unique_dates=4, is_false=False):
    """

    :return:
    """

    if not is_false:
        result = {}

        for k, v in value.items():
            if k not in result:
                result[k] = {}
            for k1, v1 in v.items():
                if k1 not in result[k]:
                    result[k][k1] = []
                for combination in v1:
                    if combination['unique_dates'] >= min_unique_dates and combination['precision'] > 0.95:
                        result[k][k1].append((combination['tree_threshold'], combination['cha_zhi_threshold'], combination['abs_threshold']))

        final_result = {}
        for k1 in result[list(result.keys())[0]]:
            common_combinations = set(result[list(result.keys())[0]][k1])
            for k in result:
                # 如果result[k][k1]不存在，则赋值为[]
                if k1 not in result[k]:
                    result[k][k1] = []
                if len(result) != 6:
                    result[k][k1] = []
                common_combinations &= set(result[k][k1])

            if common_combinations:
                min_sum = float('inf')
                min_combination = None
                for combination in common_combinations:
                    sum_threshold = sum(combination)
                    if sum_threshold < min_sum:
                        min_sum = sum_threshold
                        min_combination = combination

                final_result[k1] = {'tree_threshold': min_combination[0], 'cha_zhi_threshold': min_combination[1], 'abs_threshold': min_combination[2]}
        max_score = add_score(final_result, value)
        final_result['score'] = max_score
        return final_result
    else:
        result = {}

        for key, value1 in value.items():
            if key not in result:
                result[key] = {}
            for key1, data1 in value1.items():
                if key1 not in result[key]:
                    result[key][key1] = []
                for data in data1:
                    data['false_score'] = data['precision_false'] * data['predicted_ratio_false'] * 100 if data['precision_false'] > 0.95 else 0
                    if data['unique_dates_false'] >= min_unique_dates and data['precision_false'] > 0.95:
                        result[key][key1].append((data['tree_threshold'], data['cha_zhi_threshold'], data['abs_threshold']))
            value[key] = value1

        final_result = {}
        for k1 in result[list(result.keys())[0]]:
            common_combinations = set(result[list(result.keys())[0]][k1])
            for k in result:
                # 如果result[k][k1]不存在，则赋值为[]
                if k1 not in result[k]:
                    result[k][k1] = []
                if len(result) != 6:
                    result[k][k1] = []
                common_combinations &= set(result[k][k1])

            if common_combinations:
                min_sum = float('inf')
                min_combination = None
                for combination in common_combinations:
                    sum_threshold = sum(combination)
                    if sum_threshold < min_sum:
                        min_sum = sum_threshold
                        min_combination = combination

                final_result[k1] = {'tree_threshold': min_combination[0], 'cha_zhi_threshold': min_combination[1], 'abs_threshold': min_combination[2]}

        max_score = add_score(final_result, value, is_false=True)
        final_result['false_score'] = max_score
        return final_result

def delete_model(model_name_list):
    # 将模型名称列表转换为集合，提高查找效率
    model_name_set = set(model_name_list)
    deleted_models = []  # 用于存储已删除模型的路径

    # 遍历模型路径列表
    for model_path in MODEL_PATH_LIST:
        for root, dirs, files in os.walk(model_path):
            for file_name in files:
                if file_name.endswith('joblib') and file_name in model_name_set:
                    full_path = os.path.join(root, file_name)
                    try:
                        os.remove(full_path)  # 尝试删除文件
                        deleted_models.append(full_path)  # 记录删除的模型路径
                        print(f'{file_name}已删除')
                    except Exception as e:
                        print(f'删除{file_name}时发生错误：{e}')

    # 将删除的模型路径批量写入文件，减少文件操作次数
    deleted_models_path = '../model/all_models/deleted_model.txt'
    try:
        with open(deleted_models_path, 'a') as file:
            for model_path in deleted_models:
                file.write(model_path + '\n')
    except Exception as e:
        print(f'写入删除模型日志时发生错误：{e}')

def sort_all_report():
    """
    将所有数据按照score排序，并输出到一个文件中
    """
    file_path = '../model/reports'
    all_scores = []  # 用于存储所有的scores和对应的keys
    all_scores_false = []
    good_model_list = []
    false_model_list = []

    for root, ds, fs in os.walk(file_path):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root, f)
                # fullname = '../model/reports\smote_RandomForest_origin_data_path_dir_profit_1_day_1_bad_0.3_thread_day_1_true_ratio_0.22435625692288758_class_weight_{True_ 1, False_ 1}_max_depth_30_min_samples_leaf_2_min_samples_split_2_n_estimators_250.joblib_report.json'
                with open(fullname, 'r') as file:
                    try:
                        result_dict = json.load(file)
                        temp_score = {}
                        false_temp_score = {}
                        for key, value in result_dict.items():
                            temp_score = maintain_scores(temp_score, value)
                            false_temp_score = maintain_scores(false_temp_score, value, is_false=True)
                            all_scores.append({'key': key, 'value': temp_score})
                            all_scores_false.append({'key': key, 'value': false_temp_score})
                    except Exception as e:
                        traceback.print_exc()
                        print(f'Error occurred when reading {fullname} {e}')
                        continue

    # 按照score对all_scores进行排序，score高的排在前面
    sorted_scores = sorted(all_scores, key=lambda x: x['value']['score'], reverse=True)
    sorted_all_scores_false = sorted(all_scores_false, key=lambda x: x['value']['false_score'], reverse=True)

    # 将排序后的结果输出到一个文件中
    output_filename = '../final_zuhe/other/all_model_reports_new.json'
    false_output_filename = '../final_zuhe/other/all_model_reports_false_new.json'
    # 找到sorted_all_scores_false中false_score是0的key
    for i in range(len(sorted_all_scores_false)):
        if sorted_all_scores_false[i]['value']['false_score'] == 0:
            false_model_list.append(sorted_all_scores_false[i]['key'])
    # 找到sorted_scores中score是0的key
    for i in range(len(sorted_scores)):
        if sorted_scores[i]['value']['score'] == 0:
            good_model_list.append(sorted_scores[i]['key'])
    # # 找到false_model_list和good_model_list的交集
    # intersection = list(set(false_model_list).intersection(set(good_model_list)))
    # delete_model(intersection)
    with open(false_output_filename, 'w') as outfile:
        json.dump(sorted_all_scores_false, outfile, indent=4)
    with open(output_filename, 'w') as outfile:
        json.dump(sorted_scores, outfile, indent=4)

    print(f'Results are sorted and saved to {output_filename}')


def get_model_report(abs_name, model_name):
    """
    为单个模型生成报告，并更新模型报告文件
    :param model_path: 模型路径
    :param model_name: 当前正在处理的模型名称
    """
    try:
        # 开始计时
        train_data_list = []
        train_data_path = TRAIN_DATA_PATH
        profit = int(model_name.split('profit_')[1].split('_')[0])
        day = int(model_name.split('day_')[1].split('_')[0])
        bad = float(model_name.split('bad_')[1].split('_')[0])
        # 获取所有模型的文件名
        for root, ds, fs in os.walk(train_data_path):
            for f in fs:
                if f.endswith('_data_batch_count.csv'):
                    full_name = os.path.join(root, f)
                    full_bad = float(f.split('bad_')[1].split('_')[0])
                    if f'profit_{profit}_day_{day}' in full_name and full_bad <= bad:
                        train_data_list.append(full_name)

        start_time = time.time()

        file_path_list = train_data_list
        file_path_list.append('../train_data/profit_1_day_2024_bad_0/bad_0_data_batch_count.csv')
        # file_path_list = ['../train_data/profit_1_day_1_bad_0.3/bad_0.3_data_batch_count.csv', '../train_data/profit_1_day_1_bad_0.4/bad_0.4_data_batch_count.csv', '../train_data/profit_1_day_1_bad_0.5/bad_0.5_data_batch_count.csv', '../train_data/profit_1_day_1_bad_0.6/bad_0.6_data_batch_count.csv', '../train_data/profit_1_day_1_bad_0.7/bad_0.7_data_batch_count.csv']

        report_path = os.path.join(MODEL_REPORT_PATH, model_name + '_report.json')
        result_dict = {}
        # 加载已有报告
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    result_dict = json.load(f)
            except json.JSONDecodeError:
                result_dict = {}
        # print(f"为模型生成报告 {model_name}...")
        thread_ratio = 0.95
        new_temp_dict = {}

        try:
            model = load(abs_name)
        except Exception as e:
            if os.path.exists(abs_name):
                # 删除损坏的模型文件
                os.remove(abs_name)
                print(f"模型 {model_name} 加载失败，跳过。")
            print(f"模型 {model_name} 不存在，跳过。")
            return
        print(f"模型 {model_name} 加载成功。")
        # 提取模型的天数阈值
        thread_day = int(model_name.split('thread_day_')[1].split('_')[0])
        abs_threshold_values = np.arange(0.5, 1, 0.05)
        tree_values = np.arange(0.5, 1, 0.05)
        cha_zhi_values = np.arange(0.01, 1, 0.05)
        flag = False
        for file_path in file_path_list:
            # 判断result_dict[model_name][file_path]是否存在，如果存在则跳过
            if model_name in result_dict and file_path in result_dict[model_name]:
                if result_dict[model_name][file_path] != {}:
                    new_temp_dict[file_path] = result_dict[model_name][file_path]
                    # print(f"模型 {model_name} 对于文件 {file_path} 的报告已存在，跳过。")
                    continue
            flag = True
            temp_dict_list = []
            data = pd.read_csv(file_path, low_memory=False)
            signal_columns = [column for column in data.columns if '信号' in column]
            X_test = data[signal_columns]
            key_name = f'后续{thread_day}日最高价利润率'
            y_test = data[key_name] >= profit
            total_samples = len(y_test)

            n_trees = len(model.estimators_)
            tree_preds = np.array([tree.predict_proba(X_test) for tree in model.estimators_])

            for tree_threshold in tree_values:
                # 计算预测概率大于阈值的次数，判断为True
                true_counts = np.sum(tree_preds[:, :, 1] > tree_threshold, axis=0)
                # 计算预测概率大于阈值的次数，判断为False
                false_counts = np.sum(tree_preds[:, :, 0] > tree_threshold, axis=0)
                if true_counts.size == 0 and false_counts.size == 0:
                    continue

                # 计算大于阈值判断为True的概率
                true_proba = true_counts / n_trees
                # 计算大于阈值判断为False的概率
                false_proba = false_counts / n_trees
                proba_df = np.column_stack((false_proba, true_proba))
                y_pred_proba = proba_df
                for abs_threshold in abs_threshold_values:
                    temp_dict = {}
                    high_confidence_true = (y_pred_proba[:, 1] > abs_threshold)
                    high_confidence_false = (y_pred_proba[:, 0] > abs_threshold)
                    # 判断是否个数都是0
                    if np.sum(high_confidence_true) == 0 and np.sum(high_confidence_false) == 0:
                        continue

                    selected_true = high_confidence_true & y_test
                    selected_data = data[high_confidence_true]  # 使用布尔索引选择满足条件的数据行
                    unique_dates = selected_data['日期'].unique()  # 获取不重复的日期值
                    precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(high_confidence_true) > 0 else 0
                    predicted_true_samples = np.sum(high_confidence_true)

                    selected_false = high_confidence_false & ~y_test
                    selected_data_false = data[high_confidence_false]  # 使用布尔索引选择满足条件的数据行
                    unique_dates_false = selected_data_false['日期'].unique()  # 获取不重复的日期值
                    precision_false = np.sum(selected_false) / np.sum(high_confidence_false) if np.sum(high_confidence_false) > 0 else 0
                    predicted_false_samples_false = np.sum(high_confidence_false)


                    temp_dict['tree_threshold'] = float(tree_threshold)
                    temp_dict['cha_zhi_threshold'] = 0
                    temp_dict['abs_threshold'] = float(abs_threshold)  # 确保阈值也是原生类型
                    temp_dict['unique_dates'] = len(unique_dates.tolist())  # 将不重复的日期值转换为列表
                    temp_dict['precision'] = precision
                    temp_dict['predicted_true_samples'] = int(predicted_true_samples)
                    temp_dict['total_samples'] = int(total_samples)  # 确保转换为Python原生int类型
                    temp_dict['predicted_ratio'] = predicted_true_samples / total_samples if total_samples > 0 else 0

                    temp_dict['unique_dates_false'] = len(unique_dates_false.tolist())  # 将不重复的日期值转换为列表
                    temp_dict['precision_false'] = precision_false
                    temp_dict['predicted_false_samples_false'] = int(predicted_false_samples_false)
                    temp_dict['predicted_ratio_false'] = predicted_false_samples_false / total_samples if total_samples > 0 else 0
                    temp_dict['score'] = precision * temp_dict['predicted_ratio'] * 100 if precision > thread_ratio else 0

                    temp_dict_list.append(temp_dict)

                for cha_zhi_threshold in cha_zhi_values:
                    temp_dict = {}

                    # 计算概率差异：正类概率 - 负类概率
                    proba_diff = y_pred_proba[:, 1] - y_pred_proba[:, 0]

                    # 判断概率差异是否大于阈值
                    high_confidence_diff = (proba_diff > cha_zhi_threshold)
                    proba_diff_neg = y_pred_proba[:, 0] - y_pred_proba[:, 1]
                    high_confidence_diff_neg = (proba_diff_neg > cha_zhi_threshold)
                    if np.sum(high_confidence_diff) == 0 and np.sum(high_confidence_diff_neg) == 0:
                        continue

                    # 使用高置信差异的布尔索引选择满足条件的数据行
                    selected_data_diff = data[high_confidence_diff]
                    unique_dates_diff = selected_data_diff['日期'].unique()

                    # 选出实际为True的样本，且其概率差异大于阈值
                    selected_true_diff = high_confidence_diff & y_test
                    precision_diff = np.sum(selected_true_diff) / np.sum(high_confidence_diff) if np.sum(
                        high_confidence_diff) > 0 else 0
                    predicted_true_samples_diff = np.sum(high_confidence_diff)

                    # 与之前保持字段名称一致
                    temp_dict['tree_threshold'] = float(tree_threshold)
                    temp_dict['cha_zhi_threshold'] = float(cha_zhi_threshold)
                    temp_dict['abs_threshold'] = 0
                    temp_dict['unique_dates'] = len(unique_dates_diff.tolist())  # 将不重复的日期值转换为列表
                    temp_dict['precision'] = precision_diff
                    temp_dict['predicted_true_samples'] = int(predicted_true_samples_diff)
                    temp_dict['total_samples'] = int(total_samples)  # 确保转换为Python原生int类型
                    temp_dict['predicted_ratio'] = predicted_true_samples_diff / total_samples if total_samples > 0 else 0

                    # 另一情况：负类概率 - 正类概率 大于阈值的统计
                    # 注意：此处逻辑同上，差异在于方向相反

                    selected_data_diff_neg = data[high_confidence_diff_neg]
                    unique_dates_diff_neg = selected_data_diff_neg['日期'].unique()
                    selected_false_diff_neg = high_confidence_diff_neg & ~y_test
                    precision_diff_neg = np.sum(selected_false_diff_neg) / np.sum(high_confidence_diff_neg) if np.sum(
                        high_confidence_diff_neg) > 0 else 0
                    predicted_false_samples_diff_neg = np.sum(high_confidence_diff_neg)

                    # 为了区分，我们添加一个新的字段后缀 "_neg" 来表示这是负类概率 - 正类概率 大于阈值的情况
                    temp_dict['unique_dates_false'] = len(unique_dates_diff_neg.tolist())
                    temp_dict['precision_false'] = precision_diff_neg
                    temp_dict['predicted_false_samples_false'] = int(predicted_false_samples_diff_neg)
                    temp_dict['predicted_ratio_false'] = predicted_false_samples_diff_neg / total_samples if total_samples > 0 else 0
                    temp_dict['score'] = temp_dict['precision'] * temp_dict[
                        'predicted_ratio'] * 100 if temp_dict['precision'] > thread_ratio else 0

                    temp_dict_list.append(temp_dict)

            y_pred_proba = model.predict_proba(X_test)
            for abs_threshold in abs_threshold_values:
                temp_dict = {}
                high_confidence_true = (y_pred_proba[:, 1] > abs_threshold)
                high_confidence_false = (y_pred_proba[:, 0] > abs_threshold)
                if np.sum(high_confidence_true) == 0 and np.sum(high_confidence_false) == 0:
                    continue

                selected_true = high_confidence_true & y_test
                selected_data = data[high_confidence_true]  # 使用布尔索引选择满足条件的数据行
                unique_dates = selected_data['日期'].unique()  # 获取不重复的日期值
                precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(
                    high_confidence_true) > 0 else 0
                predicted_true_samples = np.sum(high_confidence_true)

                selected_false = high_confidence_false & ~y_test
                selected_data_false = data[high_confidence_false]  # 使用布尔索引选择满足条件的数据行
                unique_dates_false = selected_data_false['日期'].unique()  # 获取不重复的日期值
                precision_false = np.sum(selected_false) / np.sum(high_confidence_false) if np.sum(
                    high_confidence_false) > 0 else 0
                predicted_false_samples_false = np.sum(high_confidence_false)

                temp_dict['tree_threshold'] = 0
                temp_dict['cha_zhi_threshold'] = 0
                temp_dict['abs_threshold'] = float(abs_threshold)  # 确保阈值也是原生类型
                temp_dict['unique_dates'] = len(unique_dates.tolist())  # 将不重复的日期值转换为列表
                temp_dict['precision'] = precision
                temp_dict['predicted_true_samples'] = int(predicted_true_samples)
                temp_dict['total_samples'] = int(total_samples)  # 确保转换为Python原生int类型
                temp_dict['predicted_ratio'] = predicted_true_samples / total_samples if total_samples > 0 else 0

                temp_dict['unique_dates_false'] = len(unique_dates_false.tolist())  # 将不重复的日期值转换为列表
                temp_dict['precision_false'] = precision_false
                temp_dict['predicted_false_samples_false'] = int(predicted_false_samples_false)
                temp_dict[
                    'predicted_ratio_false'] = predicted_false_samples_false / total_samples if total_samples > 0 else 0
                temp_dict['score'] = precision * temp_dict['predicted_ratio'] * 100 if precision > thread_ratio else 0

                temp_dict_list.append(temp_dict)

            for cha_zhi_threshold in cha_zhi_values:
                temp_dict = {}

                # 计算概率差异：正类概率 - 负类概率
                proba_diff = y_pred_proba[:, 1] - y_pred_proba[:, 0]

                # 判断概率差异是否大于阈值
                high_confidence_diff = (proba_diff > cha_zhi_threshold)
                proba_diff_neg = y_pred_proba[:, 0] - y_pred_proba[:, 1]
                high_confidence_diff_neg = (proba_diff_neg > cha_zhi_threshold)
                if np.sum(high_confidence_diff) == 0 and np.sum(high_confidence_diff_neg) == 0:
                    continue

                # 使用高置信差异的布尔索引选择满足条件的数据行
                selected_data_diff = data[high_confidence_diff]
                unique_dates_diff = selected_data_diff['日期'].unique()

                # 选出实际为True的样本，且其概率差异大于阈值
                selected_true_diff = high_confidence_diff & y_test
                precision_diff = np.sum(selected_true_diff) / np.sum(high_confidence_diff) if np.sum(
                    high_confidence_diff) > 0 else 0
                predicted_true_samples_diff = np.sum(high_confidence_diff)

                # 与之前保持字段名称一致
                temp_dict['tree_threshold'] = 0
                temp_dict['cha_zhi_threshold'] = float(cha_zhi_threshold)
                temp_dict['abs_threshold'] = 0
                temp_dict['unique_dates'] = len(unique_dates_diff.tolist())  # 将不重复的日期值转换为列表
                temp_dict['precision'] = precision_diff
                temp_dict['predicted_true_samples'] = int(predicted_true_samples_diff)
                temp_dict['total_samples'] = int(total_samples)  # 确保转换为Python原生int类型
                temp_dict['predicted_ratio'] = predicted_true_samples_diff / total_samples if total_samples > 0 else 0

                # 另一情况：负类概率 - 正类概率 大于阈值的统计
                # 注意：此处逻辑同上，差异在于方向相反

                selected_data_diff_neg = data[high_confidence_diff_neg]
                unique_dates_diff_neg = selected_data_diff_neg['日期'].unique()
                selected_false_diff_neg = high_confidence_diff_neg & ~y_test
                precision_diff_neg = np.sum(selected_false_diff_neg) / np.sum(high_confidence_diff_neg) if np.sum(
                    high_confidence_diff_neg) > 0 else 0
                predicted_false_samples_diff_neg = np.sum(high_confidence_diff_neg)

                # 为了区分，我们添加一个新的字段后缀 "_neg" 来表示这是负类概率 - 正类概率 大于阈值的情况
                temp_dict['unique_dates_false'] = len(unique_dates_diff_neg.tolist())
                temp_dict['precision_false'] = precision_diff_neg
                temp_dict['predicted_false_samples_false'] = int(predicted_false_samples_diff_neg)
                temp_dict[
                    'predicted_ratio_false'] = predicted_false_samples_diff_neg / total_samples if total_samples > 0 else 0
                temp_dict['score'] = temp_dict['precision'] * temp_dict[
                    'predicted_ratio'] * 100 if temp_dict['precision'] > thread_ratio else 0

                temp_dict_list.append(temp_dict)

            # 按照score排序
            temp_dict_list = sorted(temp_dict_list, key=lambda x: x['score'], reverse=True)
            new_temp_dict[file_path] = temp_dict_list

        result_dict[model_name] = new_temp_dict
        # 将结果保存到文件
        if flag:
            with open(report_path, 'w') as f:
                json.dump(result_dict, f)
            # 结束计时
            end_time = time.time()
            print(f"模型报告已生成: {model_name}，耗时 {end_time - start_time:.2f} 秒\n\n")

        return result_dict
    except Exception as e:
        print(f"生成报告时出现异常: {e}")
        return {}

def build_models():
    """
    训练所有模型
    """
    origin_data_path_list = ['../train_data/profit_1_day_1_bad_0.3/bad_0.3_data_batch_count.csv']
    for origin_data_path in origin_data_path_list:
        train_all_model(origin_data_path, profit=1, thread_day_list=[1], is_skip=True)

def build_models1():
    """
    训练所有模型
    """
    origin_data_path_list = ['../daily_all_100_bad_0.5/1.txt']
    for origin_data_path in origin_data_path_list:
        train_all_model(origin_data_path, [2], is_skip=True)

def build_models2():
    """
    训练所有模型
    """
    origin_data_path_list = ['../daily_all_100_bad_0.0/1.txt']
    for origin_data_path in origin_data_path_list:
        train_all_model(origin_data_path, [2], is_skip=True)

def statistics_first_matching(daily_precision, min_unique_dates=4):
    """
    返回第一个满足条件的统计信息:
    :param daily_precision: 每日精确度数据
    :param min_unique_dates: 最小的独特日期数
    :return: 第一个满足条件的统计结果
    """
    min_day_list = sorted(set([0] + [precision['count'] for precision in daily_precision.values()]))
    max_false_count = 0
    kui_count = 0
    for min_day in min_day_list:
        best_date_count = middle_date_count = single_date_count = 0
        single_num_count = single_true_num_count = 0

        for precision in daily_precision.values():
            if precision['count'] > min_day:
                single_date_count += 1
                single_num_count += precision['count']
                single_true_num_count += precision['true_count']
                best_date_count += precision['precision'] >= 0.9
                middle_date_count += precision['precision'] >= 0.9 or precision['false_count'] <= 1
                # if precision['precision'] < 0.9:
                #     max_false_count = max(max_false_count, precision['false_count'])
                kui = 3 * precision['false_count'] - precision['true_count']
                if kui > 0:
                    kui_count += 1
                max_false_count = kui_count

        if single_date_count >= min_unique_dates and best_date_count / single_date_count >= 0.9:
            return {
                'true_num_count': single_true_num_count,
                'num_count': single_num_count,
                'true_num_score': single_true_num_count / single_num_count if single_num_count else 0,
                'best_date_count': best_date_count,
                'middle_date_count': middle_date_count,
                'date_count': single_date_count,
                'best_date_score': best_date_count / single_date_count if single_date_count else 0,
                'middle_date_score': middle_date_count / single_date_count if single_date_count else 0,
                'min_day': min_day,
                'max_false_count': max_false_count,
            }

    # 如果没有找到任何满足条件的结果
    return {}

def statistics_optimized(daily_precision, min_unique_dates=4):
    """
    优化后的统计每个min_day下面的一些统计信息函数:
    :param daily_precision: 每日精确度数据
    :param min_unique_dates: 最小的独特日期数
    :return: 统计结果
    """
    # 初始化结果字典和min_day列表
    result = {}
    min_day_list = set([0])

    # 单次遍历以构建min_day_list并计算统计信息
    for date, precision in daily_precision.items():
        min_day_list.add(precision['count'])
        # 直接在这里计算可以减少后面的遍历

    # 遍历min_day_list计算统计信息
    for min_day in min_day_list:
        best_date_count = middle_date_count = single_date_count = 0
        single_num_count = single_true_num_count = 0

        for precision in daily_precision.values():
            if precision['count'] > min_day:
                single_date_count += 1
                single_num_count += precision['count']
                single_true_num_count += precision['true_count']
                best_date_count += precision['precision'] >= 0.9
                middle_date_count += precision['precision'] >= 0.9 or precision['false_count'] <= 1

        temp_dict = {
            'true_num_count': single_true_num_count,
            'num_count': single_num_count,
            'true_num_score': single_true_num_count / single_num_count if single_num_count else 0,
            'best_date_count': best_date_count,
            'middle_date_count': middle_date_count,
            'date_count': single_date_count,
            'best_date_score': best_date_count / single_date_count if single_date_count else 0,
            'middle_date_score': middle_date_count / single_date_count if single_date_count else 0,
            'min_day': min_day,
        }
        result[min_day] = temp_dict

    # 过滤并排序结果
    result = {k: v for k, v in result.items() if v['best_date_score'] >= 0.9 and v['date_count'] >= min_unique_dates}
    result = dict(sorted(result.items()))
    # 访问result的第一个元素
    if result:
        result = list(result.values())[0]
    else:
        result = {}
    return result


def deal_reports(report_list, min_unique_dates=3, sort_key='date_count', sort_key2='date_count'):
    result_list = []
    for report in report_list:
        result = {}
        result['tree_threshold'] = report['tree_threshold']
        result['cha_zhi_threshold'] = report['cha_zhi_threshold']
        result['abs_threshold'] = report['abs_threshold']
        daily_precision = report['daily_precision']
        statistics_result = statistics_first_matching(daily_precision, min_unique_dates=min_unique_dates)
        if statistics_result:
            for k, v in statistics_result.items():
                result[k] = v
            result['true_stocks_set'] = report['true_stocks_set']
            result_list.append(result)
    # 先按照date_count降序排序，再按照middle_score降序排序
    result_list = sorted(result_list, key=lambda x: (x[sort_key], x[sort_key2]), reverse=True)
    return result_list

def sort_new():
    """
    新的排序结果
    :return:
    """
    file_path = '../model/reports'
    good_model_list = []
    sort_key = 'best_date_count'
    sort_key2 = 'best_date_count'
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            if f.endswith('.json'):
                # f = 'RandomForest_origin_data_path_dir_profit_1_day_1_bad_0.4_thread_day_1_true_ratio_0.3107309813750629_max_depth_100_min_samples_leaf_1_min_samples_split_2_n_estimators_600.joblib_report.json'
                fullname = os.path.join(root, f)
                # fullname = '../model/reports/RandomForest_origin_data_path_dir_profit_1_day_1_bad_0.3_thread_day_1_true_ratio_0.2236552287808054_max_depth_400_min_samples_leaf_3_min_samples_split_2_n_estimators_250.joblib_report.json'
                with open(fullname, 'r') as file:
                    try:
                        result_dict = json.load(file)
                        for model_name, report_value in result_dict.items():
                            # if 'thread_day_1' not in model_name:
                            #     continue
                            for test_data_path, detail_report in report_value.items():
                                if 'all' in test_data_path:
                                    temp_dict = {}
                                    max_score = 0
                                    date_count = 0
                                    precision = 0
                                    abs_threshold = 0
                                    min_day = 0
                                    true_stocks_set = []
                                    temp_dict['model_name'] = model_name
                                    for this_key, report_list in detail_report.items():
                                        result_list = deal_reports(report_list, sort_key=sort_key, sort_key2=sort_key2)
                                        # 获取result_list中每个元素的abs_threshold，和precision，生成一个新的dict
                                        precision_dict = {}
                                        for result in report_list:
                                            temp_abs_threshold = round(result['abs_threshold'], 2)
                                            temp_precision = result['precision']
                                            precision_dict[temp_abs_threshold] = temp_precision

                                        if result_list:
                                            if result_list[0][sort_key] > max_score:
                                                max_score = result_list[0][sort_key]
                                                date_count = result_list[0]['date_count']
                                                precision = result_list[0]['true_num_score']
                                                abs_threshold = result_list[0]['abs_threshold']
                                                min_day = result_list[0]['min_day']
                                                true_stocks_set = result_list[0]['true_stocks_set']
                                                max_false_count = result_list[0]['max_false_count']
                                    temp_dict['max_score'] = max_score
                                    temp_dict['date_count'] = date_count
                                    temp_dict['precision'] = precision
                                    temp_dict['abs_threshold'] = abs_threshold
                                    temp_dict['min_day'] = min_day
                                    temp_dict['true_stocks_set'] = true_stocks_set
                                    temp_dict['max_false_count'] = max_false_count
                                    temp_dict['precision_dict'] = precision_dict
                                    good_model_list.append(temp_dict)
                    except Exception as e:
                        traceback.print_exc()
                        pass
    # 将good_model_list先按照date_count降序排序再按照max_score降序排序
    good_model_list = sorted(good_model_list, key=lambda x: (-x['abs_threshold'], x['date_count']), reverse=True) # 10个好的 最大46
    with open('/mnt/w/project/python_project/a_auto_trade/final_zuhe/other/all_model_reports_cuml.json', 'w') as outfile:
        json.dump(good_model_list, outfile, indent=4)



def move_model_report(model_name_list):
    # 将model_name_list每个元素增加一个后缀_report.json
    model_name_list = [model_name + '_report.json' for model_name in model_name_list]

    # 获取MODEL_REPORT_PATH下面的所有json
    file_path = MODEL_REPORT_PATH
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root, f)
                if f in model_name_list:
                    new_path = os.path.join(DELETED_MODEL_REPORT_PATH, f)
                    shutil.move(fullname, new_path)


def delete_bad_model():
    with open('/mnt/w/project/python_project/a_auto_trade/final_zuhe/other/all_model_reports_cuml.json', 'r') as file:
        all_model_reports = json.load(file)
    model_name_list = []
    all_rf_model_list = load_rf_model_new(0, True)
    good_model_name = [model['model_name'] for model in all_rf_model_list]
    for model in all_model_reports:
        if model['model_name'] not in good_model_name:
            model_name_list.append(model['model_name'])
    delete_model(model_name_list)
    move_model_report(model_name_list)

def find_small_abs(thread_count=100, need_filter=True):
    file_path_list = ['../model/deleted_reports', '../model/reports']
    good_model_list = []
    for file_path in file_path_list:
        for root, ds, fs in os.walk(file_path):
            for f in fs:
                if f.endswith('.json'):
                    # f = 'RandomForest_origin_data_path_dir_profit_1_day_1_bad_0.4_thread_day_1_true_ratio_0.3107309813750629_max_depth_100_min_samples_leaf_1_min_samples_split_2_n_estimators_600.joblib_report.json'
                    fullname = os.path.join(root, f)
                    with open(fullname, 'r') as file:
                        try:
                            result_dict = json.load(file)
                            for model_name, report_value in result_dict.items():
                                # if 'thread_day_1' not in model_name:
                                #     continue
                                for test_data_path, detail_report in report_value.items():
                                    if 'all' in test_data_path:
                                        abs_detail = detail_report['tree_0_abs_1'][-1]
                                        if abs_detail['unique_dates'] > thread_count:
                                            temp_dict = {}
                                            temp_dict['model_name'] = model_name
                                            temp_dict['abs_threshold'] = abs_detail['abs_threshold']
                                            temp_dict['unique_dates'] = abs_detail['unique_dates']
                                            temp_dict['true_stocks_set'] = abs_detail['true_stocks_set']
                                            good_model_list.append(temp_dict)
                        except Exception as e:
                            pass

    # 将good_model_list写入文件
    good_model_list = sorted(good_model_list, key=lambda x: (-x['abs_threshold'], x['unique_dates']), reverse=True) # 10个好的 最大46
    exist_stocks = set()
    result_dict_list = []
    for sorted_scores in good_model_list:
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
                print(f"模型 {sorted_scores['model_name']} 已经有相似的模型，跳过。")
                continue
            exist_stocks = exist_stocks | current_stocks
            sorted_scores['true_stocks_set'] = []
            result_dict_list.append(sorted_scores)
    with open('/mnt/w/project/python_project/a_auto_trade/final_zuhe/other/all_reports_cuml.json', 'w') as outfile:
        json.dump(result_dict_list, outfile, indent=4)

# 将build_models和get_all_model_report用两个进程同时执行
if __name__ == '__main__':
    find_small_abs()
    # sort_new()
    # all_rf_model_list = load_rf_model_new(100, True, need_balance=False, model_max_size=500) # 200:326 100:938 0:998
    # delete_bad_model()