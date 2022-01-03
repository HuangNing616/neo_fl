from party.client import PartyA, PartyB
from party.server import PartyC
import numpy as np


# 纵向联邦学习LR
def vertical_logistic_regression(XA_train,XB_train,y_train,config):

    # Step1 将各参与方进行初始化
    party_A = PartyA(XA_train, config)
    print("参与方A 已成功初始化.")
    party_B = PartyB(XB_train, y_train, config)
    print("参与方B 已成功初始化.")
    party_C = PartyC(XA_train.shape, XB_train.shape, config)
    print("参与方C 已成功初始化.")

    # 将各个参与方进行连接, 本质上在每个客户端上, 将其他客户端的键值对{其他客户端别名:
    # 其他客户端对象}保存到 other_party 这个属性中
    party_A.connect("neofl_B", party_B)
    party_A.connect("neofl_C", party_C)

    party_B.connect("neofl_A", party_A)
    party_B.connect("neofl_C", party_C)

    party_C.connect("neofl_A", party_A)
    party_C.connect("neofl_B", party_B)


    ## 开始训练, 根据配置的迭代次数
    for i in range(config['n_iter']):

        # Step2 生成秘钥并发送给客户端A和B,每次迭代都产生不同的秘钥
        party_C.send_public_key("neofl_A", "neofl_B")

        # Step3 互相传递加密部分
        party_A.send_encrypted_items("neofl_B")
        party_B.send_encrypted_items("neofl_A")

        # Step4 计算各自的梯度部分，并将其发送给partyC
        party_A.send_encrypt_gradient("neofl_C")
        party_B.send_encrypt_gradient("neofl_C")

        # Step5 接收A,B发来的加密梯度,并进行解密
        end = party_C.send_decrypt_gradient("neofl_A", "neofl_B")

        party_A.update_model_theta()
        party_B.update_model_theta()

        if end == True:
            print("提前终止迭代")
            break

        print(f"====== 第{i+1}轮迭代完成 =========== ")

    print("所有迭代全部完成 Success!!!")
    return party_C.loss, party_A.theta, party_B.theta


# 标准损失逻辑回归
def normal_logistic_regression(X_train, y_train, config):

    # 参数初始化
    theta = np.zeros(X_train.shape[1])
    normal_loss_list = []

    ## 开始训练, 根据配置的迭代次数
    for i in range(config['n_iter']):

        # 计算梯度
        dl=0
        for j in range(X_train.shape[0]):
            tmp = 1/(1+np.exp(y_train[j]*X_train[j,:].dot(theta)))
            dl += -tmp * y_train[j]*X_train[j,:]

        # 计算损失(去掉惩罚项)
        normal_loss = np.sum(np.log(1+np.exp(-y_train*X_train.dot(theta))))/X_train.shape[0]
        normal_loss_list.append(normal_loss)

        # 更新theta
        theta = theta - config['eta'] * dl / X_train.shape[0]

    return normal_loss_list, theta


# 泰勒近似型逻辑回归
def taylor_logistic_regression(X_train, y_train, config):

    # 参数初始化
    theta = np.zeros(X_train.shape[1])# 将逻辑回归的训练参数theta初始化为1
    # 记录泰勒近似的损失函数
    taylor_loss_list = []

    ## 开始训练, 根据配置的迭代次数
    for i in range(config['n_iter']):

        # 计算梯度
        tmp = 0.25*X_train.dot(theta) - 0.5*y_train
        dl = X_train.T.dot(tmp)

        # 计算损失(去掉惩罚项)
        taylor_loss = np.sum(-0.5*y_train * X_train.dot(theta) + 0.125 * (X_train.dot(theta)*X_train.dot(theta)))/X_train.shape[0]  + math.log(2)
        taylor_loss_list.append(taylor_loss)

        # 更新参数
        if len(taylor_loss_list)>1 and taylor_loss_list[-2]-taylor_loss_list[-1]<0.000001:
            print(f"前后两次迭代损失差为{taylor_loss_list[-2]-taylor_loss_list[-1]}, 提前终止迭代")
            break

        # 更新theta
        theta = theta - config['eta'] * dl / X_train.shape[0]

    # 获取最终的训练参数theta
    return taylor_loss_list, theta