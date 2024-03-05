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
import time
from multiprocessing import Process, Pool

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

D_MODEL_PATH = 'D:\model/all_models'
G_MODEL_PATH = 'G:\model/all_models'
MODEL_PATH = '../model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
MODEL_REPORT_PATH = '../model/all_model_reports'
MODEL_OTHER = '../model/other'


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

def sort_all_report():
    """
    将所有数据按照score排序，并输出到一个文件中
    """
    file_path = '../model/all_model_reports'
    all_scores = []  # 用于存储所有的scores和对应的keys
    good_model_list = []

    for root, ds, fs in os.walk(file_path):
        for f in fs:
            if f.endswith('.json'):
                max_threshold = 0
                fullname = os.path.join(root, f)
                with open(fullname, 'r') as file:
                    try:
                        result_dict = json.load(file)

                        for key, value in result_dict.items():
                            score = 1
                            # 假设value是一个字典，其中包含一个或多个评估指标，包括score
                            for k, v in value.items():
                                if v[0]['threshold'] > max_threshold:
                                    max_threshold = v[0]['threshold']
                                if v and 'score' in v[0]:  # 确保v是一个列表，并且第一个元素是一个字典且包含score
                                    if v[0]['predicted_true_samples'] < 4:
                                        score = 0
                                    score *= v[0]['score']  # 累乘score
                            # score = 0
                            # # 假设value是一个字典，其中包含一个或多个评估指标，包括score
                            # for k, v in value.items():
                            #     if v and 'score' in v[0]:  # 确保v是一个列表，并且第一个元素是一个字典且包含score
                            #         score += v[0]['score']  # 累乘score
                            all_scores.append((key, score, max_threshold))
                    except json.JSONDecodeError:
                        print(f'Error occurred when reading {fullname}')
                        continue

    # 按照score对all_scores进行排序，score高的排在前面
    sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)

    # 将排序后的结果输出到一个文件中
    output_filename = '../final_zuhe/other/all_model_reports.json'
    with open(output_filename, 'w') as outfile:
        json.dump(sorted_scores, outfile, indent=4)

    print(f'Results are sorted and saved to {output_filename}')

def get_model_report(model_path, model_name):
    """
    为单个模型生成报告，并更新模型报告文件
    :param model_path: 模型路径
    :param model_name: 当前正在处理的模型名称
    """
    try:
        # 开始计时
        start_time = time.time()
        file_path_list = ['../daily_all_2024/1.txt', '../daily_all_100_bad_0.5/1.txt', '../daily_all_100_bad_0.3/1.txt']

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
            model = load(os.path.join(model_path, model_name))
        except Exception as e:
            if os.path.exists(os.path.join(model_path, model_name)):
                # 删除损坏的模型文件
                os.remove(os.path.join(model_path, model_name))
                print(f"模型 {model_name} 加载失败，跳过。")
            print(f"模型 {model_name} 不存在，跳过。")
            return
        # 提取模型的天数阈值
        thread_day = int(model_name.split('thread_day_')[1].split('_')[0])
        threshold_values = np.arange(0.5, 1, 0.05)
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
            signal_columns = [column for column in data.columns if 'signal' in column]
            X_test = data[signal_columns]
            y_test = data['Days Held'] <= thread_day
            total_samples = len(y_test)
            y_pred_proba = model.predict_proba(X_test)
            for threshold in threshold_values:
                temp_dict = {}
                high_confidence_true = (y_pred_proba[:, 1] > threshold)
                selected_true = high_confidence_true & y_test
                selected_data = data[high_confidence_true]  # 使用布尔索引选择满足条件的数据行
                unique_dates = selected_data['日期'].unique()  # 获取不重复的日期值
                precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(high_confidence_true) > 0 else 0
                predicted_true_samples = np.sum(high_confidence_true)

                temp_dict['threshold'] = float(threshold)  # 确保阈值也是原生类型
                temp_dict['unique_dates'] = len(unique_dates.tolist())  # 将不重复的日期值转换为列表
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

def get_all_model_report():
    """
    使用多进程获取所有模型的报告。
    """
    while True:
        model_list = []
        model_path = MODEL_PATH
        for model_path in MODEL_PATH_LIST:
            model_list += [model for model in os.listdir(model_path) if model.endswith('.joblib')]

        # 使用进程池来并行处理每个模型的报告生成
        with Pool(10) as p:
            p.starmap(get_model_report, [(model_path, model_name) for model_name in model_list])
        # time.sleep(60)  # 每隔一天重新生成一次报告


# 将build_models和get_all_model_report用两个进程同时执行
if __name__ == '__main__':
    p1 = Process(target=build_models)
    # p11 = Process(target=build_models1)
    # p12 = Process(target=build_models2)
    # p2 = Process(target=get_all_model_report)

    p1.start()
    # p2.start()
    # p11.start()
    # p12.start()

    p1.join()
    # p1.join()
    # p2.join()
    # p12.join()

    # good_model_list = []
    # output_filename = '../temp/all_model_reports.json'
    # # 加载output_filename，找到最好的模型
    # with open(output_filename, 'r') as file:
    #     sorted_scores = json.load(file)
    #     for model_name, score, threshold in sorted_scores:
    #         if score > 0.8:
    #             good_model_list.append((model_name, score, threshold))
    # sort_all_report()