from config import Config
from utils import create_dataloader
from train import create_model, train

import torch

config = Config()
train_iter, dev_iter, test_iter = create_dataloader(config)
model = create_model(config)
src_label = {'label': config.LABEL,
             'src': config.SRC,
             }

import dill

with open('src_label.pkl', 'wb') as f:
    dill.dump(src_label, f)

epochs = config.epochs
optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
train(model, train_iter, dev_iter, optimizer, epochs, scheduler)
