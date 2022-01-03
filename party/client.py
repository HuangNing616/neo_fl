from party.root import Party
import numpy as np


# 参与方A: 在训练过程中，仅提供特征
class PartyA(Party):

    # 初始化，当需要继承父类构造函数中的内容，且子类需要在父类的基础上补充时，使用super().__init__()方法。
    def __init__(self, XA, config):
        super().__init__(config)
        self.X = XA
        self.theta = np.zeros(XA.shape[1])

    # A: Step3部分，计算encrypt的中间项, 并发送给PartyB
    def send_encrypted_items(self, client_B_name):

        # 获取对应的密钥信息
        partyA_info = self.data
        assert "public_key" in partyA_info.keys(), "Error: 在Step2中send_public_key部分, PartyA没有成功接收到ClientC的 'public_key'. "
        public_key = partyA_info['public_key']

        # 计算中间项 XA * thetaA 并进行加密
        za = np.dot(self.X, self.theta) # 得到n*1的向量
        ua = 0.25 * za
        za_square = za ** 2

        # 更新自己的加密信息
        encrypted_ua = np.asarray([public_key.encrypt(x) for x in ua])  # 针对向量中的每个元素进行加密
        partyA_info.update({"encrypted_ua": encrypted_ua})              # 将加密后的中间项1/4*XA * thetaA保存到partyA_info

        # 求平方后，将对应的dict发送给B，为啥需要za_square这个平方项目
        encrypted_za_square = np.asarray([public_key.encrypt(x) for x in za_square])
        data_to_B = {"encrypted_ua": encrypted_ua, "encrypted_za_square": encrypted_za_square}
        self.send_data(data_to_B, self.other_party[client_B_name])

    # A: step4部分，计算加密的梯度，并发送给partyC
    def send_encrypt_gradient(self, party_C_name):

        # 获取自身的基本信息：包括秘钥，加密的中间项
        dt = self.data
        assert "encrypted_ub" in dt.keys(), "Error: 在PartyB的Step3中 PartyA没有成功接收到PartyB发过来的'encrypted_ub'."

        encrypted_ub = dt['encrypted_ub']
        encrypted_u = dt['encrypted_ua'] + encrypted_ub

        # 加密后的损失函数梯度为, 数乘不会影响同态性，其中第二项是损失函数的惩罚项，用的self.theta是前一轮的参数, 后续可以删除self.config['lambda'] * self.theta 这一项
        # 这个惩罚项后续删除！！！+ self.config['lambda'] * self.theta
        encrypted_dl_a = self.X.T.dot(encrypted_u)

        # 新增同维度随机项，并将其也用同样的方式进行加密
        mask = np.random.rand(len(encrypted_dl_a))
        public_key = dt['public_key'] # 走到这一步，一定能取到秘钥
        encrypted_mask = np.asarray([public_key.encrypt(x) for x in mask])  # 针对向量中的每个元素进行加密

        # 增加随机项后的梯度为
        encrypted_masked_dl_a = encrypted_dl_a + encrypted_mask

        # 将随机项保存，用于后续的解码
        dt.update({"mask": mask})
        data_to_C = {'encrypted_masked_dl_a': encrypted_masked_dl_a}

        # 将加密后的带有随机项的梯度发送给partyC
        self.send_data(data_to_C, self.other_party[party_C_name])

    # partyA: 接收解密的梯度，更新参数，并开启下一轮迭代
    def update_model_theta(self):

        # 获取解密的梯度
        dt = self.data
        assert "decrypted_masked_dl_a" in dt.keys(), "在 partyC的step5中，PartyA没有成功接收到来自partyC的解密的梯度decrypted_masked_dl_a."
        decrypted_masked_dl_a = dt['decrypted_masked_dl_a']
        dl_a = decrypted_masked_dl_a - dt['mask']

        # 更新模型参数theta = self.theta - .config["eta"](学习率) / n * dl_a
        self.theta = self.theta - self.config["eta"] * dl_a / self.X.shape[0]


# 参与方B: 在训练过程中，提供特征和标签
class PartyB(Party):

    def __init__(self, XB, y, config):
        super().__init__(config)
        self.X = XB
        self.y = y
        self.theta = np.zeros(XB.shape[1])# 将逻辑回归的训练参数theta初始化为1
        self.data = {}

    # Step3 接收PartyA中的加密项, 并将自己的加密项发送给A
    def send_encrypted_items(self, client_A_name):

        dt = self.data
        assert "public_key" in dt.keys(), "Error: 在Step2中send_public_key部分, PartyB没有成功接收到ClientC的 'public_key'."
        public_key = dt['public_key']

        zb = np.dot(self.X, self.theta)
        ub = 0.25*zb - 0.5 * self.y
        encrypted_ub = np.asarray([public_key.encrypt(x) for x in ub])

        # 更新自己的加密信息
        dt.update({"encrypted_ub": encrypted_ub})
        dt.update({"zb": zb}) # 用于后续计算加密的loss

        # 构造发送给A的数据
        data_to_A= {"encrypted_ub": encrypted_ub}
        self.send_data(data_to_A, self.other_party[client_A_name])

    # B: step4部分，计算加密的梯度，并发送给partyC
    def send_encrypt_gradient(self,client_C_name):

        # 不考虑惩罚项+ self.config['lambda'] * self.theta
        dt = self.data
        assert "encrypted_ua" in dt.keys(), "Error: 在PartyA的Step3中 PartyB没有成功接收到PartyA发过来的'encrypted_ua'."
        encrypted_ua = dt['encrypted_ua']
        encrypted_u = encrypted_ua + dt['encrypted_ub']
        encrypted_dl_b = self.X.T.dot(encrypted_u)

        # 新增同维度随机项(认为也是加密项)
        mask = np.random.rand(len(encrypted_dl_b))

        # 增加随机项后的梯度为
        public_key = dt['public_key'] # 走到这一步，一定能取到秘钥
        encrypted_mask = np.asarray([public_key.encrypt(x) for x in mask])  # 针对向量中的每个元素进行加密
        encrypted_masked_dl_b = encrypted_dl_b + encrypted_mask

        # 更新PartyB中的随机数
        dt.update({"mask": mask})

        assert "encrypted_za_square" in dt.keys(), "Error: 在PartyA的Step3中 PartyB没有成功接收到PartyA发过来的'encrypted_za_square'"
        encrypted_z = 4 * encrypted_ua + dt['zb'] # 这一项是4*0.25*XA*thetaA + XB*thetaB = XA*thetaA + XB*thetaB
        # 计算加密后的Loss
        encrypted_loss = np.sum(-0.5*self.y*encrypted_z + 0.125*dt["encrypted_za_square"] + 0.125*dt["zb"] * (8*encrypted_ua + dt["zb"]) )

        # 将加密后的B梯度以及加密后的loss保存成dict，并将其发送给clientC
        data_to_C = {"encrypted_masked_dl_b": encrypted_masked_dl_b, "encrypted_loss": encrypted_loss}
        self.send_data(data_to_C, self.other_party[client_C_name])

    # partyB: 接收解密的梯度，更新参数，并开启下一轮迭代
    def update_model_theta(self):

        # 获取解密的梯度
        dt = self.data
        assert "decrypted_masked_dl_b" in dt.keys(), "在 partyC的step5中，PartyB没有成功接收到来自partyC的解密的梯度decrypted_masked_dl_b."
        decrypted_masked_dl_b = dt['decrypted_masked_dl_b']
        dl_b = decrypted_masked_dl_b - dt['mask']

        # 更新模型参数theta = self.theta - .config["eta"](学习率) /
        self.theta = self.theta - self.config["eta"] * dl_b / self.X.shape[0]