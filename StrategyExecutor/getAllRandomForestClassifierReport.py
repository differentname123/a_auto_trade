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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Pool

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from StrategyExecutor.common import low_memory_load

D_MODEL_PATH = 'D:\model/all_models'
G_MODEL_PATH = 'G:\model/all_models'
MODEL_PATH = '../model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
MODEL_REPORT_PATH = '../model/reports'
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


def predict_proba(tree, X_test):
    return tree.predict_proba(X_test)


def predict_proba_subset(trees, X_test):
    preds = [predict_proba(tree, X_test) for tree in trees]
    return np.stack(preds)


def split_estimators(estimators, n_splits):
    k, m = divmod(len(estimators), n_splits)
    return [estimators[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_splits)]


# 这是一个新定义的具名函数，用于替代之前的lambda表达式
def predict_subset(trees_X_test):
    trees, X_test = trees_X_test
    return predict_proba_subset(trees, X_test)


def parallel_predict_proba(estimators, X_test, n_jobs=10):
    start = time.time()
    splitted_estimators = split_estimators(estimators, n_jobs)

    # 将X_test作为参数一起打包，以便传递给predict_subset函数
    trees_X_test_pairs = [(trees, X_test) for trees in splitted_estimators]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(predict_subset, trees_X_test_pairs))

    combined_results = np.concatenate(results, axis=0)
    print(f"并行预测耗时: {time.time() - start:.2f}秒")
    return combined_results

def predict_in_batches(estimators, X_test, batch_size=100000):
    # 划分X_test为多个批次，每个批次最多包含10万行
    batches = [X_test[i:i + batch_size] for i in range(0, X_test.shape[0], batch_size)]
    results = []
    print(f"开始预测，共{len(batches)}批次...")

    for batch in batches:
        # 对每个批次进行并行预测
        batch_result = parallel_predict_proba(estimators, batch)
        results.append(batch_result)

    # 合并所有批次的结果
    final_results = np.concatenate(results, axis=1)
    return final_results

def get_model_report(abs_name, model_name, need_reverse=False):
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
                    if f'profit_{profit}_day_{day}' in full_name and full_bad <= 0.5:
                        train_data_list.append(full_name)

        start_time = time.time()

        file_path_list = []

        file_path_list.append('../train_data/profit_1_day_2024_bad_0/bad_0_data_batch_count.csv')
        file_path_list.extend(train_data_list)
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
        this_key_map = {
            'tree_1_abs_1': True,'tree_0_abs_1':True,
            'tree_1_cha_zhi_1':True, 'tree_0_cha_zhi_1':True
        }
        for file_path in file_path_list:
            # 如果this_key_map的值都是False，则跳出循环
            if not any(this_key_map.values()):
                print(f" 分数全为0 对于文件 {file_path}所有this_key全为false 模型 {model_name}")
                break
            file_start_time = time.time()
            # 判断result_dict[model_name][file_path]是否存在，如果存在则跳过
            if model_name in result_dict and file_path in result_dict[model_name]:
                if result_dict[model_name][file_path] != {}:
                    new_temp_dict[file_path] = result_dict[model_name][file_path]
                    # print(f"模型 {model_name} 对于文件 {file_path} 的报告已存在，跳过。")
                    continue
            flag = True
            temp_dict_result = {}
            start = time.time()
            data = low_memory_load(file_path)
            # # 将data取前1000行
            # data = data.head(1000)
            print(f"加载数据 {file_path} 耗时: {time.time() - start:.2f}秒")
            signal_columns = [column for column in data.columns if '信号' in column]
            X_test = data[signal_columns]
            key_name = f'后续{thread_day}日最高价利润率'
            y_test = data[key_name] >= profit
            total_samples = len(y_test)
            n_trees = len(model.estimators_)
            if this_key_map['tree_1_abs_1'] or this_key_map["tree_1_cha_zhi_1"]:
                start = time.time()
                tree_preds = predict_in_batches(model.estimators_, X_test)
                end_time = time.time()
                print(f"预测耗时: {end_time - start:.2f}秒 ({n_trees}棵树)")

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
                        this_key = 'tree_1_abs_1'
                        if not this_key_map[this_key]:
                            break

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
                        temp_dict['false_score'] = precision_false * temp_dict['predicted_ratio_false'] * 100 if precision_false > thread_ratio else 0
                        if this_key not in temp_dict_result:
                            temp_dict_result[this_key] = []
                        temp_dict_result[this_key].append(temp_dict)

                    for cha_zhi_threshold in cha_zhi_values:
                        temp_dict = {}
                        this_key = 'tree_1_cha_zhi_1'
                        if not this_key_map[this_key]:
                            break
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
                        precision_false = np.sum(selected_false_diff_neg) / np.sum(high_confidence_diff_neg) if np.sum(
                            high_confidence_diff_neg) > 0 else 0
                        predicted_false_samples_diff_neg = np.sum(high_confidence_diff_neg)

                        # 为了区分，我们添加一个新的字段后缀 "_neg" 来表示这是负类概率 - 正类概率 大于阈值的情况
                        temp_dict['unique_dates_false'] = len(unique_dates_diff_neg.tolist())
                        temp_dict['precision_false'] = precision_false
                        temp_dict['predicted_false_samples_false'] = int(predicted_false_samples_diff_neg)
                        temp_dict['predicted_ratio_false'] = predicted_false_samples_diff_neg / total_samples if total_samples > 0 else 0
                        temp_dict['score'] = temp_dict['precision'] * temp_dict[
                            'predicted_ratio'] * 100 if temp_dict['precision'] > thread_ratio else 0
                        temp_dict['false_score'] = precision_false * temp_dict[
                            'predicted_ratio_false'] * 100 if precision_false > thread_ratio else 0

                        if this_key not in temp_dict_result:
                            temp_dict_result[this_key] = []
                        temp_dict_result[this_key].append(temp_dict)
            else:
                print('分数全为0 tree_1_abs_1 and tree_1_cha_zhi_1 is False')
            if this_key_map['tree_0_abs_1'] or this_key_map["tree_0_cha_zhi_1"]:
                y_pred_proba = model.predict_proba(X_test)
                for abs_threshold in abs_threshold_values:
                    this_key = 'tree_0_abs_1'
                    if not this_key_map[this_key]:
                        break
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
                    temp_dict['false_score'] = precision_false * temp_dict[
                        'predicted_ratio_false'] * 100 if precision_false > thread_ratio else 0
                    if this_key not in temp_dict_result:
                        temp_dict_result[this_key] = []
                    temp_dict_result[this_key].append(temp_dict)

                for cha_zhi_threshold in cha_zhi_values:
                    temp_dict = {}
                    this_key = 'tree_0_cha_zhi_1'
                    if not this_key_map[this_key]:
                        break
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
                    precision_false = np.sum(selected_false_diff_neg) / np.sum(high_confidence_diff_neg) if np.sum(
                        high_confidence_diff_neg) > 0 else 0
                    predicted_false_samples_diff_neg = np.sum(high_confidence_diff_neg)

                    # 为了区分，我们添加一个新的字段后缀 "_neg" 来表示这是负类概率 - 正类概率 大于阈值的情况
                    temp_dict['unique_dates_false'] = len(unique_dates_diff_neg.tolist())
                    temp_dict['precision_false'] = precision_false
                    temp_dict['predicted_false_samples_false'] = int(predicted_false_samples_diff_neg)
                    temp_dict[
                        'predicted_ratio_false'] = predicted_false_samples_diff_neg / total_samples if total_samples > 0 else 0
                    temp_dict['score'] = temp_dict['precision'] * temp_dict[
                        'predicted_ratio'] * 100 if temp_dict['precision'] > thread_ratio else 0
                    temp_dict['false_score'] = precision_false * temp_dict[
                        'predicted_ratio_false'] * 100 if precision_false > thread_ratio else 0

                    if this_key not in temp_dict_result:
                        temp_dict_result[this_key] = []
                    temp_dict_result[this_key].append(temp_dict)
            else:
                print('分数全为0 tree_0_abs_1 and tree_0_cha_zhi_1 is False')
            # 按照score排序
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
            new_temp_dict[file_path] = temp_dict_result
            print(f"耗时 {time.time() - file_start_time:.2f}秒 模型 {model_name} 对于文件 {file_path} 的报告已生成")

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
        # 获取所有模型的文件名
        for root, ds, fs in os.walk(model_path):
            for f in fs:
                full_name = os.path.join(root, f)
                if f.endswith('.joblib'):
                    model_list.append((full_name, f))
        for full_name, model_name in model_list:
            get_model_report(full_name, model_name)

        # # 使用进程池来并行处理每个模型的报告生成
        # with Pool(2) as p:
        #     p.starmap(get_model_report, [(full_name, model_name) for full_name, model_name in model_list])
        # time.sleep(60)  # 每隔一天重新生成一次报告

def get_all_model_report1():
    """
    使用多进程获取所有模型的报告。
    """
    while True:
        model_list = []
        model_path = MODEL_PATH
        # 获取所有模型的文件名
        for root, ds, fs in os.walk(model_path):
            for f in fs:
                full_name = os.path.join(root, f)
                if f.endswith('.joblib'):
                    model_list.append((full_name, f))
        model_list.reverse()
        for full_name, model_name in model_list:
            get_model_report(full_name, model_name, True)

        # # 使用进程池来并行处理每个模型的报告生成
        # with Pool(2) as p:
        #     p.starmap(get_model_report, [(full_name, model_name) for full_name, model_name in model_list])
        # time.sleep(60)  # 每隔一天重新生成一次报告


# 将build_models和get_all_model_report用两个进程同时执行
if __name__ == '__main__':
    # p1 = Process(target=build_models)
    # p11 = Process(target=build_models1)
    # p12 = Process(target=build_models2)
    p2 = Process(target=get_all_model_report)
    p21 = Process(target=get_all_model_report1)

    # p1.start()
    p2.start()
    p21.start()
    # p11.start()
    # p12.start()

    # p1.join()
    # p1.join()
    p2.join()
    p21.join()
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