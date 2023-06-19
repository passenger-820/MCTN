from paras import get_parameters
import torch

args = get_parameters()

# Sum Magnitude Vector: to find the impact point L_im
def SMV(batch_data, axis_dim=1):
    if batch_data.ndim == 2:
        axis_dim = 0
    elif batch_data.ndim == 3:
        axis_dim = 1
    else:
        return "Input shape error."
    smv_pow2 = torch.pow(batch_data, 2)
    smv_sum = torch.sum(smv_pow2, dim=axis_dim)
    smv = torch.sqrt(smv_sum)
    idx = torch.argmax(smv, dim=axis_dim)
    return idx, smv, smv_sum

# get fall prediction window:
# front_len         from l to l+L
# pre_len           \vartheta
# smv_max_idx       L_im: the impact point
# rear_len          [optional][default: 0]
class GetFeaWinFromDataset():
    def __init__(self, multi_axis_dataset, pre_len=args.pre_len, front_len=args.front_len, rear_len=args.rear_len):
        acc = multi_axis_dataset[:, 0:3, :]
        smv_max_idx, smv_max, _ = SMV(acc)

        self.batch_data = multi_axis_dataset
        self.target_idx = smv_max_idx - pre_len
        self.front_len = front_len
        self.rear_len = rear_len

    def getFeaWinByTgtIdx(self):
        if self.batch_data.ndim == 2:
            if (self.front_len >= 0) and (self.rear_len >= 0) and (self.front_len <= self.target_idx) and (
                    self.target_idx + self.rear_len + 1 <= self.batch_data.shape[-1]):
                self.batch_data[..., 0:(self.front_len + self.rear_len + 1)] = self.batch_data[...,
                                                                (self.target_idx - self.front_len):(self.target_idx + self.rear_len + 1)]
            else:
                self.batch_data = torch.zeros_like(self.batch_data)
            return self.batch_data[..., 0:(self.front_len + self.rear_len + 1)]

        if self.batch_data.ndim == 3:
            for i in range(self.batch_data.shape[0]):
                if (self.front_len >= 0) and (self.rear_len >= 0) and (self.front_len <= self.target_idx[i]) and (
                        self.target_idx[i] + self.rear_len + 1 <= self.batch_data[i].shape[-1]):
                    self.batch_data[i, :, 0:(self.front_len + self.rear_len + 1)] = self.batch_data[i, :, (self.target_idx[i] - self.front_len):(
                                self.target_idx[i] + self.rear_len + 1)]
                else:
                    self.batch_data[i, ...] = torch.zeros_like(self.batch_data[i, ...])

            non_zero_rows = torch.abs(self.batch_data[:, 0, :]).sum(dim=-1) > 0
            self.batch_data = self.batch_data[non_zero_rows]
            return self.batch_data[..., 0:(self.front_len + self.rear_len + 1)]

    def getWindow(self):
        fea_window = self.getFeaWinByTgtIdx()
        return fea_window

# Features to be extracted by some methods.
class Features():
    def __init__(self, batch_data):
        self.data = batch_data
        self.batch_size = batch_data.shape[0]
        self.fea_num = batch_data.shape[1]
        self.fea_len = batch_data.shape[2]

    def SMV_A(self):  # Sum Magnitude Vector
        smv_pow2 = torch.pow(self.data[:, :3, :], 2)
        smv_sum = torch.sum(smv_pow2, dim=1)
        smv = torch.sqrt(smv_sum)
        return smv, smv_sum

    def SMV_G(self):  # Sum Magnitude Vector
        smv_pow2 = torch.pow(self.data[:, 3:6, :], 2)
        smv_sum = torch.sum(smv_pow2, dim=1)
        smv = torch.sqrt(smv_sum)
        return smv, smv_sum

    def RMS_A(self): # Root Mean Square
        smv_pow2 = torch.pow(self.data[:, :3, :], 2)
        smv_sum = torch.sum(smv_pow2, dim=1)
        rms = torch.sqrt(smv_sum/3.)
        return rms

    def HORIZONTAL_A(self): # Horizontal Component
        smv_all = torch.pow(self.data[:, :3, :], 2)
        smv_sum = torch.sum(smv_all, dim=1)
        vertical = torch.pow(self.data[:, 1, :], 2)
        horizontal = torch.sqrt(smv_sum - vertical)
        return horizontal

    def add_feature(self, batch_data, batch_feature):
        added = torch.cat([batch_data, torch.zeros((batch_data.shape[0], 1, batch_data.shape[2]))], dim=1)
        feature_splited = batch_feature.split(1, dim=0)
        for i in range(len(feature_splited)):
            added[i, batch_data.shape[1]] = feature_splited[i]
        return added

    def add_all(self, batch_data):
        smv_a, smv_sum_a = self.SMV_A()
        rms = self.RMS_A()
        h = self.HORIZONTAL_A()
        smv_g, smv_sum_g = self.SMV_G()

        tmp = self.add_feature(batch_data, smv_a)
        tmp = self.add_feature(tmp, rms)
        tmp = self.add_feature(tmp, h)
        tmp = self.add_feature(tmp, smv_g)
        return tmp