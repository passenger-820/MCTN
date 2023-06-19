import math
import time
import torch
from torch import nn
import numpy as np
from paras import get_parameters
from utils import AverageMeter, ProgressMeter
from softfall_dataset import MySoftfallData
from mobiact_dataset import MyMobiactData
from sisfall_dataset import MySisfallData

args = get_parameters()


# Transformer Encoder
class backbone(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        super(backbone, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=args.dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


# Global Average Pooling
class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()

    def forward(self, x):
        return x.mean(dim=-2)


# Dataset
my_dataset = MySoftfallData()
# my_dataset = MyMobiactData()
# my_dataset = MySisfallData()

# train / test  split
train_size = int(0.6 * len(my_dataset))  # 0.7 for SisFall
test_size = len(my_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset=my_dataset, lengths=[train_size, test_size])

# DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=0, pin_memory=True, drop_last=False)

# MCTN Model
model = nn.Sequential(
    # Transformer Encoder
    backbone(d_model=args.d_model, nhead=args.n_head),
    # LFL
    nn.Linear(args.d_model, args.projected_dim),
    nn.LayerNorm(args.projected_dim),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.1),
    # GPL
    GAP(),
    # BPL
    nn.Linear(args.projected_dim, args.predicted_dim)
)

print(model)

# Experimental settings
device = torch.device('cuda:0')
net = model.to(device)
epochs = args.epochs
init_lr = args.lr


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


optimizer = torch.optim.SGD(net.parameters(), init_lr,
                            momentum=0.9,
                            weight_decay=1e-4)
criteon = nn.CrossEntropyLoss().to(device)

# Trainer
for epoch in range(epochs):
    adjust_learning_rate(optimizer, init_lr, epoch, epochs)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    net.train()
    end = time.time()

    # Training with training set
    for step, (train_signal, train_label) in enumerate(train_loader):
        # train_signal = train_signal[:, 0:6, ...].to(device) # a,g
        train_signal = train_signal.to(device)  # a,g,m / a,g,a

        train_label = train_label.to(device)

        train_logits = net(train_signal)
        train_loss = criteon(train_logits, train_label)
        losses.update(train_loss.item(), train_signal.size(0))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if step % 10 == 0:
            progress.display(step)

    net.eval()
    eval_start_time = time.time()

    # Testing with test set
    with torch.no_grad():
        test_loss = 0
        correct = 0

        for (test_signal, test_label) in test_loader:
            # test_signal = test_signal[:, 0:6, ...].to(device) # a,g
            test_signal = test_signal.to(device)  # a,g,m / a,g,a
            test_label = test_label.to(device)

            test_logits = net(test_signal)
            test_loss += criteon(test_logits, test_label)
            pred = test_logits.argmax(dim=1)
            correct += pred.eq(test_label).float().sum().item()

        eval_end_time = time.time() - eval_start_time

        print('\n{} Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            eval_end_time, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))