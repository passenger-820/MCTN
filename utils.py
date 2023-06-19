from paras import get_parameters
import torch
import torch.nn as nn
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

args = get_parameters()

# Tool for recording experiments
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Tool for recording experiments
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# Note: I have saved the original csv or txt file in the pytorch format.
# Due to the limitation of file size, it cannot be given in the supplementary materials.
class GetPthData():
    def __init__(self, pth_data_dir, file_name):
        self.dir = pth_data_dir
        self.file_name = file_name

    def get_raw(self):
        full_dir = os.path.join(self.dir, self.file_name)
        raw_data = torch.load(full_dir)
        return torch.transpose(raw_data,1,2)

    def down_sample(self, step=args.step_down):
        data = self.get_raw()
        return data[:, :, ::step]

# PE
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# CE
def channelEmbedding(modal_num, batch_data):
    modal_types = torch.zeros(3)
    for i in range(modal_num - 1):
        modal_types = torch.cat([modal_types, torch.ones(3) + i])
    modal_embedded = torch.cat([modal_types.view(1, modal_num*3, 1).expand(batch_data.shape[0], -1, -1), batch_data], dim=2)
    return modal_embedded
# CE-Ablation: -1 for all channels
def channelEmbeddingAblation(modal_num, batch_data):
    modal_types = torch.zeros(3)-1
    for i in range(modal_num - 1):
        modal_types = torch.cat([modal_types, torch.zeros(3)-1])
    modal_embedded = torch.cat([modal_types.view(1, modal_num*3, 1).expand(batch_data.shape[0], -1, -1), batch_data], dim=2)
    return modal_embedded