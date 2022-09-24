import os
import torch


class Config:
    def __init__(self):
        self.SRC = None
        self.LABEL = None

        self.train_path = '../Bi-LSTM-CRF/business-privacy-identify1/Business_Privacy_Identify/data/clue/train.json'
        self.dev_path = '../Bi-LSTM-CRF/business-privacy-identify1/Business_Privacy_Identify/data/clue/dev.json'
        self.test_path = '../Bi-LSTM-CRF/business-privacy-identify1/Business_Privacy_Identify/data/clue/test.json'

        # todo 修改数据集路径
        self.train_path = 'C:/Users/Heyix/Desktop/WorkSpace/bilstm-crf/Bi-LSTM-CRF/Business_Privacy_Identify/data/clue/train.json'
        self.dev_path = 'C:/Users/Heyix/Desktop/WorkSpace/bilstm-crf/Bi-LSTM-CRF/Business_Privacy_Identify/data/clue/dev.json'
        self.test_path = 'C:/Users/Heyix/Desktop/WorkSpace/bilstm-crf/Bi-LSTM-CRF/Business_Privacy_Identify/data/clue/test.json'

        abspath = os.path.abspath('')
        # 把相对路径改成绝对路径
        #         self.test_path, self.dev_path, self.test_path = map(lambda x: os.path.join(abspath, x),
        #                                                             [self.test_path, self.dev_path, self.test_path])

        self.fix_length = 50
        self.batch_size = 100
        self.embedding_dim = 768

        self.hid_dim = 300
        self.n_layers = 2
        self.dropout = 0.1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 30
        self.lr = 0.00005
        self.momentum = 0.95
