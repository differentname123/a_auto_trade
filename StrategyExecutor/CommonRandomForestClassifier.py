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

from imblearn.over_sampling import SMOTE
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

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

def load_rf_model_new(date_count_threshold=100):
    """
    加载随机森林模型
    :param model_path:
    :return:
    """
    all_rf_model_list = []
    output_filename = '../final_zuhe/other/all_model_reports_cuml.json'
    model_file_list = []
    for model_path in MODEL_PATH_LIST:
        # 获取model_path下的所有文件的全路径，如果是目录还需要继续递归
        for root, dirs, files in os.walk(model_path):
            for file in files:
                model_file_list.append(os.path.join(root, file))
    # 加载output_filename，找到最好的模型
    with open(output_filename, 'r') as file:
        sorted_scores_list = json.load(file)
        for sorted_scores in sorted_scores_list:
            model_name = sorted_scores['model_name']
            model_file_path = None
            for model_path in model_file_list:
                if model_name in model_path:
                    model_file_path = model_path
                    break
            if model_file_path is not None:
                if sorted_scores['date_count'] > date_count_threshold:
                    other_dict = sorted_scores
                    other_dict['model_path'] = model_file_path
                    all_rf_model_list.append(other_dict)
    print(f"加载了 {len(all_rf_model_list)} 个模型")
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

def get_proba_data(data, rf_classifier):
    signal_columns = [column for column in data.columns if '信号' in column]
    # 获取data中在signal_columns中的列
    X = data[signal_columns]
    # 获取data中去除signal_columns中的列
    X1 = data.drop(signal_columns, axis=1)
    y_pred_proba = rf_classifier.predict_proba(X)
    return y_pred_proba, X1

def get_proba_data_tree(data, rf_classifier):
    signal_columns = [column for column in data.columns if '信号' in column]
    # 获取data中在signal_columns中的列
    X = data[signal_columns]
    # 获取data中去除signal_columns中的列
    X1 = data.drop(signal_columns, axis=1)
    tree_preds = np.array([tree.predict_proba(X) for tree in rf_classifier.estimators_])
    return tree_preds, X1

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
    try:
        rf_model = load(rf_model_map['model_path'])

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
        print(f"模型 {model_name} 耗时 {elapsed_time}")

def get_all_good_data_with_model_name_list(data, plus_threshold=0.05):
    all_rf_model_list = load_rf_model(need_load=False)
    print(f"加载了 {len(all_rf_model_list)} 个模型 plus_threshold={plus_threshold}")

    # 使用Pool对象来并行处理
    with Pool(processes=multiprocessing.cpu_count() - 15) as pool:  # 可以调整processes的数量以匹配你的CPU核心数量
        results = pool.starmap(process_model, [(model, data, plus_threshold) for model in all_rf_model_list])

    # 过滤掉None结果并合并DataFrame
    all_selected_samples = pd.concat([res for res in results if res is not None])

    return all_selected_samples

def get_all_good_data_with_model_name_list_new(data, date_count_threshold=50):
    all_rf_model_list = load_rf_model_new(date_count_threshold)
    # for model in all_rf_model_list:
    #     print(process_model_new(model, data))

    # 使用Pool对象来并行处理
    with Pool(processes=10) as pool:  # 可以调整processes的数量以匹配你的CPU核心数量
        results = pool.starmap(process_model_new, [(model, data) for model in all_rf_model_list])

    # 过滤掉None结果并合并DataFrame
    all_selected_samples = pd.concat([res for res in results if res is not None])
    all_selected_samples.to_csv(f'../temp/all_selected_samples_{date_count_threshold}.csv', index=False)
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

if __name__ == '__main__':
    # write_joblib_files_to_txt('/mnt/d/model/all_models/')
    # write_joblib_files_to_txt('/mnt/g/model/all_models/')
    # write_joblib_files_to_txt('../model/all_models/profit_1_day_1_bad_0.4')

    # origin_data_path = '../temp/real_time_price.csv'
    # data = pd.read_csv(origin_data_path, low_memory=False, dtype={'代码': str})
    # get_all_good_data(data)
    # data = pd.read_csv('../temp/all_selected_samples_0.csv', low_memory=False, dtype={'代码': str})
    all_rf_model_list = load_rf_model_new(10)
    # 将all_rf_model_list按照score升序排序
    all_rf_model_list = sorted(all_rf_model_list, key=lambda x: x['precision'])
    # data = pd.read_csv('../temp/all_selected_samples_0.csv', low_memory=False, dtype={'代码': str})
    # data = pd.read_csv('../final_zuhe/real_time/select_RF_2024-03-13_real_time.csv', low_memory=False, dtype={'代码': str})
    data = pd.read_csv('../train_data/2024_data_all.csv', low_memory=False, dtype={'代码': str})
    # data = low_memory_load('../train_data/2024_data_all.csv')
    data['日期'] = pd.to_datetime(data['日期'])
    data = data[data['日期'] >= '2024-03-01']
    # 截取data最后4000行
    # data = data.iloc[-4000:]
    # data = {}
    get_all_good_data_with_model_name_list_new(data, 50)
    # load_rf_model_new()

