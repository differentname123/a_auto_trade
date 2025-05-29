# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2024-01-30 15:24
:last_date:
    2024-01-30 15:24
:description:
这段代码构建了一个基于 GPU 加速环境下的模型训练框架，主要用于对金融或交易数据进行随机森林模型的超参搜索和训练。下面给出整体功能的概述：

环境与依赖设置

代码最开始通过 os.environ['CUDA_VISIBLE_DEVICES'] = '0' 指定使用 GPU（这里指定索引为 0 的 GPU）来加速训练。
引入了 cuml（GPU 版本的机器学习库）、cudf（GPU DataFrame 库）、sklearn.model_selection.ParameterGrid 以及 joblib（用于模型持久化）等库。
数据加载与预处理

函数 load_test_data 使用自定义的 low_memory_load 方法读取 CSV 文件，并转换为 GPU DataFrame。
该函数还选择了一些关键列（如包含“最高价利润率”的列、日期、代码及信号列）并进行了数据类型降精度以减少内存占用。
模型训练与保存

train_and_dump_model 函数完成模型的训练过程：
利用传入的训练数据（特征和标签）对随机森林分类器模型进行拟合。
模型训练结束后，通过 joblib.dump 将训练好的模型保存成指定的 .joblib 文件，并记录保存的模型名称和大小信息到一个跟踪文件中。
超参组合搜索与过滤

通过 ParameterGrid 根据预设的随机森林参数（如 n_estimators、max_depth、min_samples_split、min_samples_leaf）生成所有可能的超参组合进行遍历训练。
其中，函数 skip_params（以及辅助函数 are_other_affecting_params_same 和 are_params_triggering_problem）实现了对超参组合的过滤，避免训练那些可能导致模型体积过大或存在其他问题的参数组合。
针对不同时间窗口的模型训练

函数 train_all_model 根据传入的数据文件和不同的“天数阈值”（如 1 天、2 天、3 天）来构造标签：
根据不同的“后续N日最高价利润率”来定义正负样本（比如判断是否达到设定的利润阈值），并计算正样本的比例。
然后调用 train_models 对每种天数配置下的训练数据进行超参搜索训练。
模型管理与多线程设计

build_models 函数作为整体入口，负责遍历不同的数据文件路径，收集已存在的模型，然后依次或通过线程调用 worker（进而调用 train_all_model）来训练新的模型。
同时，代码借助多线程（尽管部分线程启动代码被注释）来提升训练效率。
模型参数解析与目标模型训练

除了批量训练外，部分函数（如 parse_filename 与 construct_final_path）通过正则表达式解析模型文件名中保存的超参信息，以便用于后续的重新训练或筛选目标模型。
train_target_model 函数读取某个文件中的目标模型列表，对这些模型进行重新训练或补充训练，确保所需模型存在。
整体流程

当直接运行该脚本时，会执行 build_models() 函数，启动整个训练流程，对指定数据目录下的 CSV 文件进行加载、特征提取、标签构造、超参搜索、模型训练、模型保存以及结果记录。

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用第二张GPU（2060），索引从0开始
import json
import re


import time
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import train_test_split
from joblib import dump
import cudf
from sklearn.model_selection import ParameterGrid
import threading

from StrategyExecutor.common import downcast_dtypes, low_memory_load

TRAIN_DATA_PATH = '/mnt/w/project/python_project/a_auto_trade/train_data'
D_MODEL_PATH = '/mnt/d/model/all_models/'
G_MODEL_PATH = '/mnt/g/model/all_models/'
F_MODEL_PATH = '/mnt/f/model/all_models/'
MODEL_PATH = '/mnt/w/project/python_project/a_auto_trade/model/all_models'
MODEL_PATH_LIST = [D_MODEL_PATH, G_MODEL_PATH, MODEL_PATH]
MODEL_OTHER = '../model/other'
MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/reports'
DELETED_MODEL_REPORT_PATH = '/mnt/w/project/python_project/a_auto_trade/model/deleted_reports'


def train_and_dump_model(clf, X_train, y_train, model_file_path, exist_model_file_path):
    """
    训练模型并保存到指定路径
    :param clf: 分类器实例
    :param X_train: 训练数据集特征
    :param y_train: 训练数据集标签
    :param model_path: 模型保存路径
    :param model_name: 模型名称
    """
    start_time = time.time()
    model_name = os.path.basename(model_file_path)
    out_put_path = model_file_path
    if not os.path.exists(os.path.dirname(out_put_path)):
        os.makedirs(os.path.dirname(out_put_path))
    print(f"当前时间{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} 开始训练模型: {model_name}")
    clf.fit(X_train, y_train)
    dump(clf, model_file_path, compress=1)
    model_name_size = round(os.path.getsize(model_file_path) / 1024 ** 2, 2)
    print(f"耗时 {time.time() - start_time} 大小为 {model_name_size} MB  模型已保存: {model_file_path} \n\n")
    with open(exist_model_file_path, 'a') as f:
        f.write(model_name + '\n')
    return model_name_size

def are_other_affecting_params_same(params1, params2, affecting_params, current_key):
    # 检查除了当前考虑的参数之外,其他影响问题的参数是否相同
    for key in affecting_params.keys():
        if key != current_key and params1[key] != int(params2[key]):
            return False
    return True

def are_params_triggering_problem(params1, params2, affecting_params):
    # 检查两个参数组合在影响问题的参数上是否满足特定条件
    for key, increases_problem in affecting_params.items():
        # 检查除了当前考虑的参数之外,其他影响问题的参数是否相同
        if not are_other_affecting_params_same(params1, params2, affecting_params, key):
            continue

        if increases_problem:
            # 如果该参数在值变大时导致问题,且当前组合的值大于等于已经导致问题的组合的值,则认为满足条件
            if params1[key] >= int(params2[key]):
                return True
        else:
            # 如果该参数在值变小时导致问题,且当前组合的值小于等于已经导致问题的组合的值,则认为满足条件
            if params1[key] <= int(params2[key]):
                return True
    return False

BIG_MODEL_SIZE_PARAMS = {
    'n_estimators': True,  # 学习率变大可能导致NAN
    'max_depth': True,  # 学习率变大可能导致NAN
    'min_samples_split': False,  # 学习率变大可能导致NAN
    'min_samples_leaf': False,  # 学习率变大可能导致NAN
}
def skip_params(params, train_results, max_model_size=200):
    big_train_results = {k: v for k, v in train_results.items() if v['model_size'] > max_model_size}
    # 根据已有的训练结果设计跳过条件
    for model_name, result in big_train_results.items():
        if are_params_triggering_problem(params, big_train_results[model_name]['params'], BIG_MODEL_SIZE_PARAMS):
            return True

def train_models(X_train, y_train, model_type, thread_day, origin_data_path_dir, report_list):
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
    true_ratio = y_train.mean()
    param_grid = {
        'RandomForest': {
            'n_estimators': [100, 200, 250, 300, 400, 500, 600],
            'max_depth': [10, 20, 30, 40, 100, 200, 400],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6]
        }
    }[model_type]
    train_results_file = os.path.join(MODEL_OTHER, f"{origin_data_path_dir}_train_result.json")
    # 尝试从文件中加载已有的训练结果
    try:
        with open(train_results_file, 'r') as f:
            train_results = json.load(f)
    except Exception as e:
        train_results = {}
        print(f"Error loading training results: {e}")
        pass

    params_list = list(ParameterGrid(param_grid))
    # random.shuffle(params_list)
    print(f"待训练的模型数量: {len(params_list)} 已有的report_list数量: {len(report_list)}")
    save_path = MODEL_PATH
    for params in params_list:

        model_name = f"{model_type}_origin_data_path_dir_{origin_data_path_dir}_thread_day_{thread_day}_true_ratio_{true_ratio}_{'_'.join([f'{key}_{value}' for key, value in params.items()])}.joblib"
        model_file_path = os.path.join(save_path, origin_data_path_dir, model_name)
        flag = False
        if model_name in report_list:
            print(f"模型已存在，跳过: {model_name}")
            continue
        if skip_params(params, train_results):
            print(f"size太大 跳过参数: {params}")
            continue
        if flag:
            continue
        exist_model_file_path = os.path.join(save_path, 'existed_model.txt')
        clf = RandomForestClassifier(**params)
        model_size = train_and_dump_model(clf, X_train, y_train, model_file_path, exist_model_file_path)
        params = {k: str(v) for k, v in params.items()}
        train_result = {}
        train_result['params'] = params
        train_result['model_size'] = model_size
        train_results[model_name] = train_result
        with open(train_results_file, 'w') as f:
            json.dump(train_results, f, indent=4)

def load_test_data(file_path):
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

def train_all_model(file_path_path, report_list, profit=1, thread_day_list=None, is_skip=True):
    """
    为file_path_path生成各种模型
    :param file_path_path: 数据集路径
    :param thread_day_list: 判断为True的天数列表
    :param is_skip: 如果已有模型，是否跳过
    """
    if thread_day_list is None:
        thread_day_list = [1, 2, 3]
    origin_data_path_dir = os.path.dirname(file_path_path)
    origin_data_path_dir = origin_data_path_dir.split('/')[-1]
    base_name = os.path.basename(file_path_path)
    # 去除base_name中的后缀
    base_name = base_name.split('.')[0]
    origin_data_path_dir = origin_data_path_dir + '_' + base_name
    print("加载数据{}...".format(file_path_path))
    final_data, X = load_test_data(file_path_path)
    ratio_result_path = os.path.join(MODEL_OTHER, origin_data_path_dir + 'ratio_result.json')
    try:
        with open(ratio_result_path, 'r') as f:
            ratio_result = json.load(f)
    except FileNotFoundError:
        ratio_result = {}
    for thread_day in thread_day_list:
        if thread_day == 3:
            key_name_2 = f'后续2日最高价利润率'
            key_name_1 = f'后续1日最高价利润率'
            y = (final_data[key_name_1] < final_data[key_name_2]) & (final_data[key_name_1] < profit)
        else:
            key_name = f'后续{thread_day}日最高价利润率'
            y = final_data[key_name] >= profit
        ratio_key = origin_data_path_dir + '_' + str(thread_day)
        if ratio_key in ratio_result:
            true_ratio = ratio_result[ratio_key]
        else:
            true_ratio = y.mean()
            ratio_result[ratio_key] = true_ratio
            with open(ratio_result_path, 'w') as f:
                json.dump(ratio_result, f)
        print(f"处理天数阈值: {thread_day}, 真实比率: {true_ratio:.4f}")
        X_train = X
        y_train = y
        train_models(X_train, y_train, 'RandomForest', thread_day, origin_data_path_dir, report_list)


def worker(origin_data_path, report_list):
    """
    作为线程工作的函数，调用 train_all_model 函数。
    """
    train_all_model(origin_data_path, report_list, profit=1, thread_day_list=[1,2], is_skip=True)

def build_models():
    """
    训练所有模型
    """
    origin_data_path_list = [
        '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_0-1_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_1-2_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_2-3_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_3-4_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_4-5_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_5-6_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_6-7_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_7-8_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_8-9_train.csv',
        # '/mnt/f/train_data/profit_1_day_1_ratio_0.25/bad_0_interval_9-10_train.csv',

        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_0-1_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_1-2_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_2-3_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_3-4_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_4-5_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_5-6_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_6-7_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_7-8_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_8-9_train.csv',
        # '/mnt/f/train_data/profit_1_day_2_ratio_0.25/bad_0_interval_9-10_train.csv',
    ]
    report_list = []
    # for root, ds, fs in os.walk(MODEL_REPORT_PATH):
    #     for f in fs:
    #         if f.endswith('report.json'):
    #             report_list.append(f.split('_report.json')[0])
    # for root, ds, fs in os.walk(DELETED_MODEL_REPORT_PATH):
    #     for f in fs:
    #         if f.endswith('report.json'):
    #             report_list.append(f.split('_report.json')[0])

    model_list = []
    for model_path in MODEL_PATH_LIST:
        # 获取所有模型的文件名
        for root, ds, fs in os.walk(model_path):
            for f in fs:
                if f.endswith('joblib'):
                    model_list.append(f)
    report_list.extend(model_list)
    report_list = list(set(report_list))

    threads = []
    for origin_data_path in origin_data_path_list:
        worker(origin_data_path, report_list)
    # get_all_model_report(500, 0.5)
    #     thread = threading.Thread(target=worker, args=(origin_data_path, report_list))
    #     threads.append(thread)
    #     thread.start()
    #
    # for thread in threads:
    #     thread.join()


def parse_filename(input_string):
    # 使用正则表达式匹配需要提取的部分
    file_name_pattern = r'(profit_\d+_day_\d+_bad_\d+\.\d+)'
    thread_day_pattern = r'thread_day_(\d+)'
    max_depth_pattern = r'max_depth_(\d+)'
    min_samples_leaf_pattern = r'min_samples_leaf_(\d+)'
    min_samples_split_pattern = r'min_samples_split_(\d+)'
    n_estimators_pattern = r'n_estimators_(\d+)'
    true_ratio_pattern = r'true_ratio_([\d\.]+)'  # 新增解析true_ratio的模式

    # 从输入字符串中搜索匹配的模式
    file_name_match = re.search(file_name_pattern, input_string)
    thread_day_match = re.search(thread_day_pattern, input_string)
    max_depth_match = re.search(max_depth_pattern, input_string)
    min_samples_leaf_match = re.search(min_samples_leaf_pattern, input_string)
    min_samples_split_match = re.search(min_samples_split_pattern, input_string)
    n_estimators_match = re.search(n_estimators_pattern, input_string)
    true_ratio_match = re.search(true_ratio_pattern, input_string)  # 新增匹配true_ratio的代码

    # 提取匹配的结果并构造输出字典
    parsed_data = {
        'file_name': file_name_match.group() if file_name_match else None,
        'thread_day': thread_day_match.group(1) if thread_day_match else None,
        'max_depth': max_depth_match.group(1) if max_depth_match else None,
        'min_samples_leaf': min_samples_leaf_match.group(1) if min_samples_leaf_match else None,
        'min_samples_split': min_samples_split_match.group(1) if min_samples_split_match else None,
        'n_estimators': n_estimators_match.group(1) if n_estimators_match else None,
        'true_ratio': true_ratio_match.group(1) if true_ratio_match else None,  # 新增true_ratio的输出
    }

    return parsed_data


def construct_final_path(file_name):
    # 从file_name中提取 bad 部分的值
    bad_value_pattern = r'bad_(\d+\.\d+)'
    bad_match = re.search(bad_value_pattern, file_name)

    # 如果匹配到了bad值，构造最终路径
    if bad_match:
        bad_value = bad_match.group(1)
        final_path = os.path.join(file_name, f'bad_{bad_value}_data_batch_count.csv')
        return final_path
    else:
        raise ValueError("The file_name does not contain a 'bad' value.")
def train_target_model():
    # 读取参数列表
    # output_filename = '/mnt/w/project/python_project/a_auto_trade/final_zuhe/other/all_reports_cuml.json'
    # with open(output_filename, 'r') as file:
    #     sorted_scores_list = json.load(file)
    # # 获取sorted_scores_list中的model_name的列表
    # model_name_list = [item['model_name'] for item in sorted_scores_list]
    model_name_list = []
    file_path = f'../final_zuhe/other/not_estimated_model_list.txt'
    with open(file_path, 'r') as lines:
        for line in lines:
            model_name_list.append(line.strip())

    # 读取模型参数
    param_list = []
    for model_name in model_name_list:
        param = parse_filename(model_name)
        param_list.append(param)
    save_path = MODEL_PATH
    model_list = []
    for model_path in MODEL_PATH_LIST:
        # 获取所有模型的文件名
        for root, ds, fs in os.walk(model_path):
            for f in fs:
                if f.endswith('joblib'):
                    model_list.append(f)

    # param_list的每个元素都是一个字典，包含了模型的参数，现在需要将param_list按照file_name的值进行分组
    file_name_dict = {}
    for param in param_list:
        file_name = param['file_name']
        if file_name in file_name_dict:
            file_name_dict[file_name].append(param)
        else:
            file_name_dict[file_name] = [param]
    for file_name, param_group in file_name_dict.items():
        # 读取数据集
        data_path = os.path.join(TRAIN_DATA_PATH, construct_final_path(file_name))
        data = cudf.read_csv(data_path)
        signal_columns = [column for column in data.columns if '信号' in column]
        X = data[signal_columns]
        X = downcast_dtypes(X)
        for params in param_group:
            # 读取标签
            thread_day = params['thread_day']
            y = data[f'后续{thread_day}日最高价利润率'] >= 1
            true_ratio = params['true_ratio']
            model_name = f"RandomForest_origin_data_path_dir_{file_name}_thread_day_{params['thread_day']}_true_ratio_{true_ratio}_max_depth_{params['max_depth']}_min_samples_leaf_{params['min_samples_leaf']}_min_samples_split_{params['min_samples_split']}_n_estimators_{params['n_estimators']}.joblib"
            if model_name in model_list:
                print(f"模型已存在，跳过: {model_name}")
                continue
            model_file_path = os.path.join(save_path, file_name, model_name)
            exist_model_file_path = os.path.join(save_path, 'existed_model.txt')
            final_param = {
                'max_depth': int(params['max_depth']),
                'min_samples_leaf': int(params['min_samples_leaf']),
                'min_samples_split': int(params['min_samples_split']),
                'n_estimators': int(params['n_estimators'])
            }
            clf = RandomForestClassifier(**final_param)
            train_and_dump_model(clf, X, y, model_file_path, exist_model_file_path)

if __name__ == '__main__':
    build_models()

    # train_target_model()