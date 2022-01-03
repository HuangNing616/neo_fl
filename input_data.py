from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


# 加载数据并分成训练集和验证集
def load_data():
    # 加载数据
    breast = load_breast_cancer()
    #     print("特征名称: ", breast.feature_names, len(breast.feature_names))
    #     print("标签取值: ", breast.target_names, len(breast.target_names))

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(breast.data, breast.target, random_state=1)

    # 数据标准化
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    return X_train, y_train, X_test, y_test


# 纵向分割数据, 将特征分配给A和B
def vertically_partition_data(X_train, X_test, A_idx, B_idx):
    """
    Vertically partition feature for party A and B
    :param X:      训练集特征数据
    :param X_test: 测试集特征数据
    :param A_idx: Party A 的特征索引
    :param B_idx: Party B 的特征索引
    :return: 分割后的训练特征数据XA_train, XB_train, 以及测试特征数据XA_test, XB_test
    """

    # 训练集分割，并在原始数组的起始部分，新增一列1
    XA_train = X_train[:, A_idx]
    XB_train = X_train[:, B_idx]

    # 测试集分割，并在原始数组的起始部分，新增一列1
    XA_test = X_test[:, A_idx]
    XB_test = X_test[:, B_idx]

    return XA_train, XB_train, XA_test, XB_test
