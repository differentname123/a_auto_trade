import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump, load
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path):
    """加载数据集并提取特征与标签"""
    data = pd.read_csv(file_path, low_memory=False)
    signal_columns = [col for col in data.columns if 'signal' in col]
    X = data[signal_columns]
    y = data['Days Held'] <= 1
    return X, y


def handle_class_imbalance(X_train, y_train):
    """使用SMOTE处理类别不平衡"""
    X_train = X_train.astype(int)
    y_train = y_train.astype(int)
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    return X_balanced, y_balanced


def train_or_load_rf_classifier(X_train, y_train, model_filename, best_params, is_skip):
    """训练或加载随机森林分类器"""
    if is_skip and os.path.exists(model_filename):
        rf_classifier = load(model_filename)
    else:
        rf_classifier = RandomForestClassifier(**best_params)
        rf_classifier.fit(X_train, y_train)
        dump(rf_classifier, model_filename)
    return rf_classifier


def calculate_detailed_votes(rf_classifier, X, threshold, min_trees_for_true):
    """计算每个样本的详细投票结果，并判断最终预测"""
    tree_preds = np.array([tree.predict_proba(X) for tree in rf_classifier.estimators_])
    votes = np.zeros((X.shape[0], 4))  # 对应 detailed_votes 的结构

    for i, tree_pred in enumerate(tree_preds):
        gt_threshold = tree_pred[:, 1] > threshold
        lt_threshold = ~gt_threshold
        true_pred = np.argmax(tree_pred, axis=1).astype(bool)

        # 累加每个条件下的投票
        votes[:, 0] += gt_threshold & true_pred
        votes[:, 1] += gt_threshold & ~true_pred
        votes[:, 2] += lt_threshold & true_pred
        votes[:, 3] += lt_threshold & ~true_pred

    # 最终预测：根据高置信度票数（大于阈值且判断为True的票数）决定
    y_pred_final = (votes[:, 0] > min_trees_for_true).astype(bool)
    return y_pred_final, votes


def calculate_detailed_votes_combined_dataframe(rf_classifier, X, thresholds, detailed_votes_path):
    """
    针对多个阈值计算每个样本的详细投票结果，并将所有结果合并为单个DataFrame。
    """
    tree_preds = np.array([tree.predict_proba(X) for tree in rf_classifier.estimators_])
    rows = []  # 用于收集每行的数据

    # 定义列名
    columns = []
    for threshold in thresholds:
        columns.extend([
            f'gt_{threshold}_true_count', f'gt_{threshold}_false_count',
            f'lt_{threshold}_true_count', f'lt_{threshold}_false_count',
            f'ratio_gt_{threshold}_true_false', f'ratio_lt_{threshold}_true_false'
        ])

    # 遍历每个样本
    for sample_index in range(X.shape[0]):
        row = []
        for threshold in thresholds:
            # 初始化计数器
            gt_true_count, gt_false_count, lt_true_count, lt_false_count = 0, 0, 0, 0

            # 遍历每棵树的预测结果
            for tree_pred in tree_preds:
                prob_true = tree_pred[sample_index, 1]  # 第二个元素的预测值（True类别）
                prob_false = tree_pred[sample_index, 0]  # 第一个元素的预测值（False类别）

                # 根据阈值判断并计数
                if prob_true > threshold:
                    gt_true_count += 1
                elif prob_true <= threshold:
                    lt_true_count += 1

                if prob_false > threshold:
                    gt_false_count += 1
                elif prob_false <= threshold:
                    lt_false_count += 1

            # 计算比值，避免除以零
            ratio_gt = gt_true_count / gt_false_count if gt_false_count > 0 else np.nan
            ratio_lt = lt_true_count / lt_false_count if lt_false_count > 0 else np.nan

            # 添加当前阈值下的结果到行数据中
            row.extend([gt_true_count, gt_false_count, lt_true_count, lt_false_count, ratio_gt, ratio_lt])

        rows.append(row)

    # 一次性创建DataFrame
    votes_combined = pd.DataFrame(rows, columns=columns)
    # 将votes_combined保存为csv文件
    votes_combined.to_csv(detailed_votes_path, index=False)
    return votes_combined


def calculate_detailed_votes_optimized(rf_classifier, X, thresholds):
    """
    使用优化的方法针对多个阈值计算每个样本的详细投票结果。
    """
    # 获取所有树的预测概率
    tree_preds = np.stack([tree.predict_proba(X) for tree in rf_classifier.estimators_])
    num_trees = len(rf_classifier.estimators_)
    # 初始化结果列表，用于存储每个阈值的计算结果
    results = []

    # 对每个阈值进行向量化的计算
    for threshold in thresholds:
        # 计算大于阈值的情况
        gt_threshold = tree_preds[:, :, 1] > threshold
        gt_true_count = np.sum(gt_threshold, axis=0)
        lt_true_count = len(rf_classifier.estimators_) - gt_true_count

        # 计算小于等于阈值的情况
        lt_threshold = tree_preds[:, :, 0] > threshold
        gt_false_count = np.sum(lt_threshold, axis=0)
        lt_false_count = len(rf_classifier.estimators_) - gt_false_count

        gt_true_count_normalized = np.round(gt_true_count / num_trees, 2)
        gt_false_count_normalized = np.round(gt_false_count / num_trees, 2)
        lt_true_count_normalized = np.round(lt_true_count / num_trees, 2)
        lt_false_count_normalized = np.round(lt_false_count / num_trees, 2)

        # 计算比值，确保分母不为零并对比值加上一个小的正数，然后保留两位小数
        ratio_gt = np.round(np.log((gt_true_count + 1e-10) / (gt_false_count + 1e-10)) / 27, 2)
        ratio_lt = np.round(np.log((lt_true_count + 1e-10) / (lt_false_count + 1e-10)) / 27, 2)

        # 替换inf值
        ratio_gt[np.isinf(ratio_gt)] = np.max(ratio_gt[~np.isinf(ratio_gt)]) if np.any(~np.isinf(ratio_gt)) else 0
        ratio_lt[np.isinf(ratio_lt)] = np.max(ratio_lt[~np.isinf(ratio_lt)]) if np.any(~np.isinf(ratio_lt)) else 0

        # 合并当前阈值的所有计算结果
        current_results = np.vstack(
            [gt_true_count_normalized, gt_false_count_normalized, lt_true_count_normalized, lt_false_count_normalized,
             ratio_gt, ratio_lt]).T
        results.append(current_results)

    # 将所有结果合并
    combined_results = np.hstack(results)

    # 将归一化后的结果转换为DataFrame
    columns = []
    for threshold in thresholds:
        columns.extend([
            f'阈值大于_{threshold}_为True比例', f'阈值大于_{threshold}_为False比例',
            f'阈值小于_{threshold}_为True比例', f'阈值小于_{threshold}_为False比例',
            f'阈值大于_{threshold}_True和False比例', f'阈值小于_{threshold}_True和False比例'
        ])

    votes_combined = pd.DataFrame(combined_results, columns=columns)

    return votes_combined

def main():
    origin_data_path = '../daily_all_2024/1.txt'
    # origin_data_path = '../daily_all_100_bad_0.5/1.txt'
    model_path = '../model/'
    detailed_votes_path = '../model/all_tree'
    best_params = {
        'n_estimators': 100,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_depth': 10
    }
    threshold = 0.7
    min_trees_for_true = 50
    is_skip = True

    # 加载数据
    X, y = load_and_prepare_data(origin_data_path)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, y_train = handle_class_imbalance(X_train, y_train)
    X_test = X
    y_test = y


    # 构建模型文件名
    model_filename = f'rf_classifier_{best_params["n_estimators"]}_{best_params["min_samples_split"]}_{best_params["min_samples_leaf"]}_{"None" if best_params["max_depth"] is None else best_params["max_depth"]}.joblib'
    model_filename = os.path.join(model_path, model_filename)

    # 训练或加载模型
    rf_classifier = train_or_load_rf_classifier(X_train, y_train, model_filename, best_params, is_skip)

    votes_combined = calculate_detailed_votes_optimized(rf_classifier, X_test, [0.5, 0.6])
    votes_combined['实际结果'] = y_test.reset_index(drop=True)

    # 计算详细投票结果
    y_pred_final, votes = calculate_detailed_votes(rf_classifier, X_test, threshold, min_trees_for_true)

    # 分析结果并保存到DataFrame
    df_votes = pd.DataFrame(votes, columns=['gt_threshold_true', 'gt_threshold_false',
                                            'lt_threshold_true', 'lt_threshold_false'])
    df_votes['实际结果'] = y_test.reset_index(drop=True)
    df_votes['预测结果'] = y_pred_final
    print(df_votes.head())

    # 可以选择保存 df_votes 到 CSV 文件
    df_votes.to_csv(os.path.join(detailed_votes_path, 'detailed_votes.csv'), index=False)

    # 计算并打印准确率
    accuracy = accuracy_score(y_test, y_pred_final)
    print(f'模型准确率: {accuracy}')

def gen_detail_votes(model_path_list, file_path, out_put_path):
    """
    生成详细的投票结果
    :param model_path: 模型地址
    :param file_path_list: 待生成文件的地址列表
    :param out_put_path: 输出文件地址
    :return:
    """
    data = pd.read_csv(file_path, low_memory=False)
    signal_columns = [col for col in data.columns if 'signal' in col]
    origin_data_path_dir = os.path.dirname(file_path)
    origin_data_path_dir = origin_data_path_dir.split('/')[-1]
    X = data[signal_columns]
    for model_path in model_path_list:
        # 使用os模块获取model_path的文件名
        model_name = os.path.basename(model_path)
        thread_day = int(model_name.split('thread_day_')[1].split('_')[0])
        rf_classifier = load(model_path)
        votes_combined = calculate_detailed_votes_optimized(rf_classifier, X, [0.5, 0.6])
        votes_combined['实际结果'] = data['Days Held'] <= thread_day
        out_put_file_name = os.path.join(out_put_path, origin_data_path_dir + '__' + model_name + '_detail_votes.csv')
        votes_combined.to_csv(out_put_file_name, index=False)
        print(f'已生成详细投票结果文件: {out_put_file_name}')


if __name__ == '__main__':
    model_path_list = ['../model/all_models/RandomForest_origin_data_path_dir_daily_all_2024_thread_day_1_true_ratio_0.7859649777844155_max_depth_10_min_samples_leaf_3_min_samples_split_5_n_estimators_600.joblib']
    file_path = '../daily_all_2024/1.txt'
    out_put_path = '../model/all_tree'
    gen_detail_votes(model_path_list, file_path, out_put_path)
    # main()
