# 各参与方的父类
class Party:

    def __init__(self, config):

        # 模型参数
        self.config = config

        # 保存各参与方的基本信息
        self.data = {}

        # 保存与其他节点的连接状况
        self.other_party = {}

    # 与其他参与方建立连接
    def connect(self, client_name, target_client):
        self.other_party[client_name] = target_client

    # 向特定参与方发送数据
    def send_data(self, data, target_client):

        target_client.data.update(data)