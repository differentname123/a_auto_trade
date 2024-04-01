import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop

# 创建一个简单的TensorFlow操作，该操作需要使用cuDNN
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU devices:", gpu_devices)
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

# 尝试执行一个操作，看是否成功
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print("cuDNN seems to be working, computed result:\n", c)

# 数据预处理和特征工程
def preprocess_data(X, y, is_jump=False):
    if is_jump:
        return X, y
    # 缺失值处理
    X.fillna(X.mean(), inplace=True)

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 特征选择
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X_scaled, y)

    # 特征降维
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_selected)

    return X_pca, y


# 定义模型构建函数
def build_model(hidden_layers, neurons, activation, l1_reg, l2_reg, dropout_rate, optimizer):
    model = Sequential()
    for i in range(hidden_layers):
        model.add(Dense(neurons[i], activation=activation, kernel_regularizer=l1(l1_reg), bias_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 模型训练和保存
def train_and_save(X_train, y_train, model_params):
    models = []

    for i, params in enumerate(model_params):
        model = build_model(**{k: v for k, v in params.items() if k != 'batch_size' and k != 'epochs'})
        model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)

        model_path = f"model_{i + 1}.keras"
        model.save(model_path)
        print(f"Model {i + 1} saved as {model_path}")

        models.append(model)

    return models


# 加载数据和模型训练
def main():
    # 假设你的数据集路径
    dataset_path = "2024_data.csv"
    data = pd.read_csv(dataset_path)
    thread_day = 1
    profit = 1
    key_name = f'后续{thread_day}日最高价利润率'
    y = data[key_name] >= profit
    signal_columns = [column for column in data.columns if '信号' in column]
    X = data[signal_columns]

    # 数据预处理和特征工程
    X_preprocessed, y_preprocessed = preprocess_data(X, y, is_jump=True)

    # 定义超参数搜索空间
    param_grid = {
        'hidden_layers': [2, 3, 4],
        'neurons': [(128, 64), (256, 128, 64), (512, 256, 128, 64)],
        'activation': ['relu', 'leakyrelu', 'elu'],
        'l1_reg': [0.001, 0.01, 0.1],
        'l2_reg': [0.001, 0.01, 0.1],
        'dropout_rate': [0.2, 0.3, 0.4],
        'optimizer': [Adam(learning_rate=0.001), Adagrad(learning_rate=0.01), RMSprop(learning_rate=0.01)],
        'batch_size': [32, 64, 128],
        'epochs': [50, 100, 150]
    }

    # 生成所有可能的超参数组合
    model_params = ParameterGrid(param_grid)

    # 模型训练和保存
    models = train_and_save(X_preprocessed, y_preprocessed, model_params)


if __name__ == '__main__':
    main()