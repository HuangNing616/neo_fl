from lr_trainer import vertical_logistic_regression, taylor_logistic_regression, normal_logistic_regression
from sklearn.metrics import roc_auc_score
from input_data import load_data, vertically_partition_data

import matplotlib.pyplot as plt
import numpy as np

# 导入数据，将数据分成训练和测试集，并标准化
X_train, y_train, X_test, y_test = load_data()
print("分割并标准化后的训练数据维度: {}行{}列".format(X_train.shape[0], X_train.shape[1]))

# 设置模型的配置参数
config = {
    'n_iter': 20,  # 迭代次数
    'eta': 0.05,  # 学习率
    'A_idx': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  # PartyA 部分的特征索引
    'B_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],                                                    # PartyB 部分的特征索引
}

# 将数据集分成 PartyA 和 PartyB 部分
XA_train, XB_train, XA_test, XB_test = vertically_partition_data(X_train, X_test, config['A_idx'], config['B_idx'])
print('PartyA部分的数据规模:{}'.format(XA_train.shape))
print('PartyB部分的数据规模:{}'.format(XB_train.shape))

# 开始训练
fl_loss, fl_theta_a, fl_theta_b = vertical_logistic_regression(XA_train, XB_train, y_train, config)
norm_loss, normal_theta = normal_logistic_regression(X_train, y_train, X_test, y_test, config)
taylor_loss, taylor_theta = taylor_logistic_regression(X_train, y_train, X_test, y_test, config)

# 绘制计算AUC以及准确率
# 针对FL的LR，计算测试集上的prob
fl_y_prob = 1/(1 + np.exp(-XA_test.dot(fl_theta_a) - XB_test.dot(fl_theta_b)))
taylor_y_prob = 1/(1 + np.exp(-X_test.dot(taylor_theta)))
normal_y_prob = 1/(1 + np.exp(-X_test.dot(normal_theta)))

fl_y_train_prob = 1/(1 + np.exp(-XA_train.dot(fl_theta_a) - XB_train.dot(fl_theta_b)))
taylor_y_train_prob = 1/(1 + np.exp(-X_train.dot(taylor_theta)))
normal_y_train_prob = 1/(1 + np.exp(-X_train.dot(normal_theta)))

print("train fl lr auc", roc_auc_score(y_train, fl_y_train_prob))
print("train taylor lr auc", roc_auc_score(y_train, taylor_y_train_prob))
print("train normal lr auc", roc_auc_score(y_train, normal_y_train_prob))

print("test fl lr auc", roc_auc_score(y_test, fl_y_prob))
print("test taylor lr auc", roc_auc_score(y_test, taylor_y_prob))
print("test normal lr auc", roc_auc_score(y_test, normal_y_prob))


# 展示拟合效果
_ = plt.plot(range(len(taylor_loss)), taylor_loss, c="blue",label="taylor lr loss")
_ = plt.plot(range(len(fl_loss)), fl_loss, label="vertical lr loss")
_ = plt.plot(range(len(norm_loss)), norm_loss, label="normal lr loss")

plt.xlabel("step")
plt.ylabel("loss value")
plt.legend(loc="upper right")

plt.show()