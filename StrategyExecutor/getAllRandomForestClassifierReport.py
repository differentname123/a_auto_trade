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

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
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


def parallel_predict_proba(estimators, X_test, process_key, n_jobs=9):
    start = time.time()
    splitted_estimators = split_estimators(estimators, n_jobs)

    # 将X_test作为参数一起打包，以便传递给predict_subset函数
    trees_X_test_pairs = [(trees, X_test) for trees in splitted_estimators]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(predict_subset, trees_X_test_pairs))

    combined_results = np.concatenate(results, axis=0)
    print(f"并行预测耗时: {time.time() - start:.2f}秒{process_key}")
    return combined_results

def predict_in_batches(estimators, X_test, batch_size=400000):
    # 划分X_test为多个批次，每个批次最多包含10万行
    batches = [X_test[i:i + batch_size] for i in range(0, X_test.shape[0], batch_size)]
    results = []
    tree_count = len(estimators)
    print(f"开始预测，共{len(batches)}批次...{len(estimators)}棵树")
    total_count= len(batches)
    count = 0
    for batch in batches:
        count += 1
        process_key = f"{tree_count}进度: {count}/{total_count}"
        # 对每个批次进行并行预测
        batch_result = parallel_predict_proba(estimators, batch, process_key)
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
        train_data_list, profit, day = get_train_data_list(model_name)
        start_time = time.time()
        file_path_list = ['../train_data/profit_1_day_2024_bad_0/bad_0_data_batch_count.csv'] + train_data_list
        report_path = os.path.join(MODEL_REPORT_PATH, model_name + '_report.json')
        result_dict = load_existing_report(report_path)
        thread_ratio = 0.95
        new_temp_dict = {}

        model = load_model(abs_name, model_name)
        if model is None:
            return

        thread_day = int(model_name.split('thread_day_')[1].split('_')[0])
        abs_threshold_values = np.arange(0.5, 1, 0.05)
        tree_values = np.arange(0.5, 1, 0.05)
        cha_zhi_values = np.arange(0.01, 1, 0.05)
        this_key_map = {
            'tree_1_abs_1': True, 'tree_0_abs_1': True,
            'tree_1_cha_zhi_1': True, 'tree_0_cha_zhi_1': True
        }

        for file_path in file_path_list:
            if not any(this_key_map.values()):
                print(f"分数全为0 对于文件 {file_path}所有this_key全为false 模型 {model_name}")
                break

            if is_report_exists(result_dict, model_name, file_path):
                new_temp_dict[file_path] = result_dict[model_name][file_path]
                continue

            file_start_time = time.time()
            temp_dict_result = {}
            data = load_data(file_path)
            signal_columns = [column for column in data.columns if '信号' in column]
            X_test = data[signal_columns]
            key_name = f'后续{thread_day}日最高价利润率'
            y_test = data[key_name] >= profit
            total_samples = len(y_test)
            n_trees = len(model.estimators_)

            if this_key_map['tree_1_abs_1'] or this_key_map["tree_1_cha_zhi_1"]:
                tree_preds = predict_tree_preds(model, X_test, n_trees)
                temp_dict_result = process_tree_preds(data, tree_preds, y_test, total_samples, tree_values, abs_threshold_values, cha_zhi_values, thread_ratio, this_key_map)

            if this_key_map['tree_0_abs_1'] or this_key_map["tree_0_cha_zhi_1"]:
                y_pred_proba = model.predict_proba(X_test)
                temp_dict_result = process_pred_proba(data, y_pred_proba, y_test, total_samples, abs_threshold_values, cha_zhi_values, thread_ratio, this_key_map, temp_dict_result)

            update_this_key_map(temp_dict_result, this_key_map, file_path, model_name)
            new_temp_dict[file_path] = temp_dict_result
            print(f"耗时 {time.time() - file_start_time:.2f}秒 模型 {model_name} 对于文件 {file_path} 的报告已生成")

            result_dict[model_name] = new_temp_dict
            save_report(report_path, result_dict)
        end_time = time.time()
        print(f"模型报告已生成: {model_name}，耗时 {end_time - start_time:.2f} 秒\n\n")

        return result_dict
    except Exception as e:
        traceback.print_exc()
        print(f"生成报告时出现异常: {e}")
        return {}

def get_train_data_list(model_name):
    train_data_list = []
    train_data_path = TRAIN_DATA_PATH
    profit = int(model_name.split('profit_')[1].split('_')[0])
    day = int(model_name.split('day_')[1].split('_')[0])
    for root, ds, fs in os.walk(train_data_path):
        for f in fs:
            if f.endswith('_data_batch_count.csv'):
                full_name = os.path.join(root, f)
                full_bad = float(f.split('bad_')[1].split('_')[0])
                if f'profit_{profit}_day_{day}' in full_name and full_bad <= 0.5:
                    train_data_list.append(full_name)
    return train_data_list, profit, day

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
        if os.path.exists(abs_name):
            os.remove(abs_name)
            print(f"模型 {model_name} 加载失败，跳过。")
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

def predict_tree_preds(model, X_test, n_trees):
    start = time.time()
    tree_preds = predict_in_batches(model.estimators_, X_test)
    end_time = time.time()
    print(f"预测耗时: {end_time - start:.2f}秒 ({n_trees}棵树)")
    return tree_preds

def process_tree_preds(data, tree_preds, y_test, total_samples, tree_values, abs_threshold_values, cha_zhi_values, thread_ratio, this_key_map):
    temp_dict_result = {}
    n_trees = tree_preds.shape[0]

    for tree_threshold in tree_values:
        true_counts = np.sum(tree_preds[:, :, 1] > tree_threshold, axis=0)
        false_counts = np.sum(tree_preds[:, :, 0] > tree_threshold, axis=0)
        if true_counts.size == 0 and false_counts.size == 0:
            break

        true_proba = true_counts / n_trees
        false_proba = false_counts / n_trees
        proba_df = np.column_stack((false_proba, true_proba))
        y_pred_proba = proba_df

        for abs_threshold in abs_threshold_values:
            this_key = 'tree_1_abs_1'
            if not this_key_map[this_key]:
                break

            temp_dict_result = process_abs_threshold(data, y_pred_proba, y_test, total_samples, abs_threshold, tree_threshold, thread_ratio, this_key, temp_dict_result)

        for cha_zhi_threshold in cha_zhi_values:
            this_key = 'tree_1_cha_zhi_1'
            if not this_key_map[this_key]:
                break

            temp_dict_result = process_cha_zhi_threshold(data, y_pred_proba, y_test, total_samples, cha_zhi_threshold, tree_threshold, thread_ratio, this_key, temp_dict_result)

    return temp_dict_result

def process_pred_proba(data, y_pred_proba, y_test, total_samples, abs_threshold_values, cha_zhi_values, thread_ratio, this_key_map, temp_dict_result):
    for abs_threshold in abs_threshold_values:
        this_key = 'tree_0_abs_1'
        if not this_key_map[this_key]:
            break

        temp_dict_result = process_abs_threshold(data, y_pred_proba, y_test, total_samples, abs_threshold, 0, thread_ratio, this_key, temp_dict_result)

    for cha_zhi_threshold in cha_zhi_values:
        this_key = 'tree_0_cha_zhi_1'
        if not this_key_map[this_key]:
            break

        temp_dict_result = process_cha_zhi_threshold(data, y_pred_proba, y_test, total_samples, cha_zhi_threshold, 0, thread_ratio, this_key, temp_dict_result)

    return temp_dict_result

def process_abs_threshold(data, y_pred_proba, y_test, total_samples, abs_threshold, tree_threshold, thread_ratio, this_key, temp_dict_result):
    high_confidence_true = (y_pred_proba[:, 1] > abs_threshold)
    high_confidence_false = (y_pred_proba[:, 0] > abs_threshold)
    if np.sum(high_confidence_true) == 0 and np.sum(high_confidence_false) == 0:
        return temp_dict_result

    selected_true = high_confidence_true & y_test
    selected_data = data[high_confidence_true]
    unique_dates = selected_data['日期'].unique()
    precision = np.sum(selected_true) / np.sum(high_confidence_true) if np.sum(high_confidence_true) > 0 else 0
    predicted_true_samples = np.sum(high_confidence_true)

    selected_false = high_confidence_false & ~y_test
    selected_data_false = data[high_confidence_false]
    unique_dates_false = selected_data_false['日期'].unique()
    precision_false = np.sum(selected_false) / np.sum(high_confidence_false) if np.sum(high_confidence_false) > 0 else 0
    predicted_false_samples_false = np.sum(high_confidence_false)

    daily_precision = {}
    daily_precision_false = {}
    if precision >= thread_ratio or precision_false >= thread_ratio:

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

    temp_dict = create_temp_dict(tree_threshold, 0, abs_threshold, unique_dates, precision, predicted_true_samples, total_samples, unique_dates_false, precision_false, predicted_false_samples_false, thread_ratio, daily_precision, daily_precision_false)
    if this_key not in temp_dict_result:
        temp_dict_result[this_key] = []
    temp_dict_result[this_key].append(temp_dict)

    return temp_dict_result

def process_cha_zhi_threshold(data, y_pred_proba, y_test, total_samples, cha_zhi_threshold, tree_threshold, thread_ratio, this_key, temp_dict_result):
    proba_diff = y_pred_proba[:, 1] - y_pred_proba[:, 0]
    high_confidence_diff = (proba_diff > cha_zhi_threshold)
    proba_diff_neg = y_pred_proba[:, 0] - y_pred_proba[:, 1]
    high_confidence_diff_neg = (proba_diff_neg > cha_zhi_threshold)
    if np.sum(high_confidence_diff) == 0 and np.sum(high_confidence_diff_neg) == 0:
        return temp_dict_result

    selected_data_diff = data[high_confidence_diff]
    unique_dates_diff = selected_data_diff['日期'].unique()
    selected_true_diff = high_confidence_diff & y_test
    precision_diff = np.sum(selected_true_diff) / np.sum(high_confidence_diff) if np.sum(high_confidence_diff) > 0 else 0
    predicted_true_samples_diff = np.sum(high_confidence_diff)

    selected_data_diff_neg = data[high_confidence_diff_neg]
    unique_dates_diff_neg = selected_data_diff_neg['日期'].unique()
    selected_false_diff_neg = high_confidence_diff_neg & ~y_test
    precision_false = np.sum(selected_false_diff_neg) / np.sum(high_confidence_diff_neg) if np.sum(high_confidence_diff_neg) > 0 else 0
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

    temp_dict = create_temp_dict(tree_threshold, cha_zhi_threshold, 0, unique_dates_diff, precision_diff, predicted_true_samples_diff, total_samples, unique_dates_diff_neg, precision_false, predicted_false_samples_diff_neg, thread_ratio, daily_precision_diff, daily_precision_diff_neg)
    if this_key not in temp_dict_result:
        temp_dict_result[this_key] = []
    temp_dict_result[this_key].append(temp_dict)

    return temp_dict_result

def create_temp_dict(tree_threshold, cha_zhi_threshold, abs_threshold, unique_dates, precision, predicted_true_samples, total_samples, unique_dates_false, precision_false, predicted_false_samples_false, thread_ratio, daily_precision, daily_precision_false):
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
    temp_dict['false_score'] = precision_false * temp_dict['predicted_ratio_false'] * 100 if precision_false > thread_ratio else 0
    temp_dict['daily_precision'] = daily_precision
    temp_dict['daily_precision_false'] = daily_precision_false
    return temp_dict

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
                if f.endswith('joblib'):
                    model_list.append((full_name, f))
        # 随机打乱model_list
        random.shuffle(model_list)
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
                if f.endswith('joblib'):
                    model_list.append((full_name, f))
        # 随机打乱model_list
        random.shuffle(model_list)
        for full_name, model_name in model_list:
            get_model_report(full_name, model_name, True)

        # # 使用进程池来并行处理每个模型的报告生成
        # with Pool(2) as p:
        #     p.starmap(get_model_report, [(full_name, model_name) for full_name, model_name in model_list])
        # time.sleep(60)  # 每隔一天重新生成一次报告


# 将build_models和get_all_model_report用两个进程同时执行
if __name__ == '__main__':
    p2 = Process(target=get_all_model_report)
    # p21 = Process(target=get_all_model_report1)
    # p211 = Process(target=get_all_model_report1)

    # p1.start()
    p2.start()
    # p21.start()
    # p211.start()

    p2.join()
    # p21.join()
    # p211.join()
