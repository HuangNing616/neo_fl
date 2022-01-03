from party.root import Party
from phe import paillier
import numpy as np
import math


# 参与方C: 在训练过程中，提供秘钥对，扮演trusted dealer(委托人)的身份
class PartyC(Party):

    def __init__(self, XA_shape, XB_shape, config):
        super().__init__(config)
        self.A_data_shape, self.B_data_shape = XA_shape, XB_shape # 保存各参与方的数据维度
        self.public_key, self.private_key = None, None           # 保存公钥和私钥
        self.loss = []                                           # 保存训练中的损失值(用Taylor展开近似）

    # Step1 发送密钥对给partyA和partyB
    def send_public_key(self, party_A_name, party_B_name):

        try:
            public_key, private_key = paillier.generate_paillier_keypair()
            self.public_key = public_key
            self.private_key = private_key

        except Exception as e:
            print("PartyC 产生密钥对的过程失败, 详细失败原因: %s" % e)

        # 将公钥保存成字典的形式
        key_to_AB = {"public_key": public_key}

        # 将公钥分别发送给PartyA和PartyB, 更新相应对象的.data属性
        self.send_data(key_to_AB, self.other_party[party_A_name])
        self.send_data(key_to_AB, self.other_party[party_B_name])
        return

        # Step5 将partyA和partyB发来的加密梯度进行解密, 并返回
    def send_decrypt_gradient(self, party_A_name, party_B_name):

        # 获取接收PartyA的加密梯度和PartyB的加密梯度
        dt = self.data
        assert "encrypted_masked_dl_a" in dt.keys() and "encrypted_masked_dl_b" in dt.keys(), "Error: 在 Step4 中 partyC 没有成功接收到来自 partyA 的'masked_dJ_a'或者来自 partyB 的'masked_dl_b'."

        encrypted_masked_dl_a = dt['encrypted_masked_dl_a']
        encrypted_masked_dl_b = dt['encrypted_masked_dl_b']

        decrypted_masked_dl_a = np.asarray([self.private_key.decrypt(x) for x in encrypted_masked_dl_a])
        decrypted_masked_dl_b = np.asarray([self.private_key.decrypt(x) for x in encrypted_masked_dl_b])

        # 将加密后的loss进行解密
        assert "encrypted_loss" in dt.keys(), "Error: 'encrypted_loss' 在Step4中 没有成功接收到来自 partyB 的'encrypted_loss'."
        encrypted_loss = dt['encrypted_loss']

        # 将加密后的loss进行解密
        # 计算解密后的loss，即除以n 再加上常数项log2
        # 不考虑惩罚项 #config['lambda']*(np.sum(decrypted_theta_a_square) + np.sum(decrypted_theta_b_square))/2*self.A_data_shape[0]
        loss = self.private_key.decrypt(encrypted_loss) / self.A_data_shape[0] + math.log(2)
        print("******本轮迭代，计算的损失loss= ", loss, "******")
        self.loss.append(loss)

        # 针对至少迭代一次的loss开始进行判断，如果两次loss相减小于0.001，或者达到最大迭代次数max_iter，那么算法停止
        if len(self.loss) > 1 and (self.loss[-2] - self.loss[-1])<0.0001:
            return True


        data_to_A = {"decrypted_masked_dl_a": decrypted_masked_dl_a}
        data_to_B = {"decrypted_masked_dl_b": decrypted_masked_dl_b}

        # 将解密后的梯度发送给 partyA 和 partyB
        self.send_data(data_to_A, self.other_party[party_A_name])
        self.send_data(data_to_B, self.other_party[party_B_name])
        return