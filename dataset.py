from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
from scipy.interpolate import griddata

class Dataset(Dataset):
    # _DATA 是形状为 (8760, 156, 151, 7) 的原始数据，共 365*24=8760 个时间片
    _DATA = np.load("/home/guoshuai/coding/PRD-MSGNN/final_dataset/prd_aq_mask_alldata.npy")
    GT = np.load("/home/guoshuai/coding/PRD-MSGNN/final_dataset/GT.npy")
    # with open('./final_dataset/station_list.pkl', 'rb') as f:
    #     station_list = pickle.load(f)

    def __init__(self, indices_path, mask_station, remain_station) -> None:
        super().__init__()
        self.indices = np.load(indices_path)
        self.mask_station = mask_station
        self.remain_station = remain_station
        # self.ctx_len = context_len
        # self.pred_len = prediction_len

    def __getitem__(self, index):
        idx = self.indices[index]
        x = self._DATA[idx, :, :, :]

        # 去掉被 mask 的站点的值
        y = self.GT[idx, :, :]
        return np.nan_to_num(x), y
        

    def __len__(self):
        return len(self.indices)