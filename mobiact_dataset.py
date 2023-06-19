from paras import get_parameters
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from features import GetFeaWinFromDataset
from utils import GetPthData, ButterWorth, PositionalEncoding, modalEmbedding, modalEmbeddingAblation


class MyMobiactData(Dataset):

    def __init__(self):
        super(MyMobiactData, self).__init__()
        args = get_parameters()
        """
        读取的数据就是batch，axis，points
        timestamp	rel_time	[acc_x	acc_y	acc_z	gyro_x	gyro_y	gyro_z]	azimuth	pitch	roll	label
        """
        BSC_Fall_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="BSC_Fall.pth").down_sample(args.mobiact_down)
        BSC_Fall_raw_downsampled = BSC_Fall_raw_downsampled[:,2:8,:]
        CHU_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="CHU_ADL.pth").down_sample(args.mobiact_down)
        CHU_ADL_raw_downsampled = CHU_ADL_raw_downsampled[:,2:8,:]
        CSI_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="CSI_ADL.pth").down_sample(args.mobiact_down)
        CSI_ADL_raw_downsampled = CSI_ADL_raw_downsampled[:,2:8,:]
        CSO_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="CSO_ADL.pth").down_sample(args.mobiact_down)
        CSO_ADL_raw_downsampled = CSO_ADL_raw_downsampled[:,2:8,:]
        FKL_Fall_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="FKL_Fall.pth").down_sample(args.mobiact_down)
        FKL_Fall_raw_downsampled = FKL_Fall_raw_downsampled[:,2:8,:]
        FOL_Fall_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="FOL_Fall.pth").down_sample(args.mobiact_down)
        FOL_Fall_raw_downsampled = FOL_Fall_raw_downsampled[:,2:8,:]
        JOG_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="JOG_ADL.pth").down_sample(args.mobiact_down)
        JOG_ADL_raw_downsampled = JOG_ADL_raw_downsampled[:,2:8,:]
        JUM_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="JUM_ADL.pth").down_sample(args.mobiact_down)
        JUM_ADL_raw_downsampled = JUM_ADL_raw_downsampled[:,2:8,:]
        SBE_Flow_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SBE_Flow_ADL.pth").down_sample(args.mobiact_down)
        SBE_Flow_ADL_raw_downsampled = SBE_Flow_ADL_raw_downsampled[:,2:8,:]
        SBW_Flow_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SBW_Flow_ADL.pth").down_sample(args.mobiact_down)
        SBW_Flow_ADL_raw_downsampled = SBW_Flow_ADL_raw_downsampled[:,2:8,:]
        SCH_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SCH_ADL.pth").down_sample(args.mobiact_down)
        SCH_ADL_raw_downsampled = SCH_ADL_raw_downsampled[:,2:8,:]
        SDL_Fall_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SDL_Fall.pth").down_sample(args.mobiact_down)
        SDL_Fall_raw_downsampled = SDL_Fall_raw_downsampled[:,2:8,:]
        SIT_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SIT_ADL.pth").down_sample(args.mobiact_down)
        SIT_ADL_raw_downsampled = SIT_ADL_raw_downsampled[:,2:8,:]
        SLH_Flow_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SLH_Flow_ADL.pth").down_sample(args.mobiact_down)
        SLH_Flow_ADL_raw_downsampled = SLH_Flow_ADL_raw_downsampled[:,2:8,:]
        SLW_Flow_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SLW_Flow_ADL.pth").down_sample(args.mobiact_down)
        SLW_Flow_ADL_raw_downsampled = SLW_Flow_ADL_raw_downsampled[:,2:8,:]
        SRH_Flow_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="SRH_Flow_ADL.pth").down_sample(args.mobiact_down)
        SRH_Flow_ADL_raw_downsampled = SRH_Flow_ADL_raw_downsampled[:,2:8,:]
        STD_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="STD_ADL.pth").down_sample(args.mobiact_down)
        STD_ADL_raw_downsampled = STD_ADL_raw_downsampled[:,2:8,:]
        STN_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="STN_ADL.pth").down_sample(args.mobiact_down)
        STN_ADL_raw_downsampled = STN_ADL_raw_downsampled[:,2:8,:]
        STU_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="STU_ADL.pth").down_sample(args.mobiact_down)
        STU_ADL_raw_downsampled = STU_ADL_raw_downsampled[:,2:8,:]
        WAL_ADL_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="WAL_ADL.pth").down_sample(args.mobiact_down)
        WAL_ADL_raw_downsampled = WAL_ADL_raw_downsampled[:,2:8,:]

        """
        feature window 以及 label
        """
        BSC_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=BSC_Fall_raw_downsampled).getWindow()
        self.label = torch.ones(BSC_Fall_fea_window.shape[0], dtype=torch.long)

        CHU_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CHU_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CHU_ADL_fea_window.shape[0], dtype=torch.long)])

        CSI_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CSI_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CSI_ADL_fea_window.shape[0], dtype=torch.long)])

        CSO_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=CSO_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(CSO_ADL_fea_window.shape[0], dtype=torch.long)])

        FKL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=FKL_Fall_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.ones(FKL_Fall_fea_window.shape[0], dtype=torch.long)])

        FOL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=FOL_Fall_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.ones(FOL_Fall_fea_window.shape[0], dtype=torch.long)])

        JOG_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=JOG_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(JOG_ADL_fea_window.shape[0], dtype=torch.long)])

        JUM_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=JUM_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(JUM_ADL_fea_window.shape[0], dtype=torch.long)])

        SBE_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SBE_Flow_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SBE_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SBW_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SBW_Flow_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SBW_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SCH_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SCH_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SCH_ADL_fea_window.shape[0], dtype=torch.long)])

        SDL_Fall_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SDL_Fall_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.ones(SDL_Fall_fea_window.shape[0], dtype=torch.long)])

        SIT_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SIT_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SIT_ADL_fea_window.shape[0], dtype=torch.long)])

        SLH_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SLH_Flow_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SLH_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SLW_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SLW_Flow_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SLW_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        SRH_Flow_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=SRH_Flow_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(SRH_Flow_ADL_fea_window.shape[0], dtype=torch.long)])

        STD_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STD_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STD_ADL_fea_window.shape[0], dtype=torch.long)])

        STN_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STN_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STN_ADL_fea_window.shape[0], dtype=torch.long)])

        STU_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=STU_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(STU_ADL_fea_window.shape[0], dtype=torch.long)])

        WAL_ADL_fea_window = GetFeaWinFromDataset(multi_axis_dataset=WAL_ADL_raw_downsampled).getWindow()
        self.label = torch.cat([self.label, torch.zeros(WAL_ADL_fea_window.shape[0], dtype=torch.long)])

        """
        原始数据和滤波后的数据分别全部concatenate在一起
        """
        self.signals = torch.cat([BSC_Fall_fea_window,CHU_ADL_fea_window,CSI_ADL_fea_window,CSO_ADL_fea_window,FKL_Fall_fea_window,FOL_Fall_fea_window,JOG_ADL_fea_window,JUM_ADL_fea_window,
                                  SBE_Flow_ADL_fea_window,SBW_Flow_ADL_fea_window,SCH_ADL_fea_window,SDL_Fall_fea_window,SIT_ADL_fea_window,SLH_Flow_ADL_fea_window,
                                  SLW_Flow_ADL_fea_window,SRH_Flow_ADL_fea_window,STD_ADL_fea_window,STN_ADL_fea_window,STU_ADL_fea_window,WAL_ADL_fea_window], dim=0)

        butterWorthFilter = ButterWorth(self.signals)
        self.signals_lowPass = torch.from_numpy(butterWorthFilter.lowPass()).to(torch.float32)

        """
        pos和modal embedding
        """
        self.signals = PositionalEncoding(d_model=args.d_pe)(self.signals)
        self.signals = modalEmbedding(modal_num=args.modal_two, batch_data=self.signals)
        # self.signals = modalEmbeddingAblation(modal_num=args.modal_two, batch_data=self.signals)

        self.signals_lowPass = PositionalEncoding(d_model=args.d_pe)(self.signals_lowPass)
        self.signals_lowPass = modalEmbedding(modal_num=args.modal_two, batch_data=self.signals_lowPass)


    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal_filtered = self.signals_lowPass[idx]
        label = self.label[idx]

        return signal, signal_filtered, label



if __name__ == '__main__':
    # print('')
    #
    mydata = MyMobiactData()
    print(mydata.signals.shape)

    print('')

    # # 可视化看下数据有没有问题
    # BSC_Fall_raw_downsampled = GetPthData(pth_data_dir=args.mobiact_dir, file_name="BSC_Fall.pth").get_raw()
    # BSC_Fall_raw_downsampled = BSC_Fall_raw_downsampled[:, 2:8, ]
    # print(BSC_Fall_raw_downsampled.shape)
    # from utils import drawAxisPicturesWithData
    # draw = drawAxisPicturesWithData(BSC_Fall_raw_downsampled, picture_title="mobiact", save_root='static/figures/')
    # draw.pltAllAxis()


