import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn.init import kaiming_normal_, constant_
from torchvision.transforms import Resize
from torch_scatter import scatter_mean
from typing import Callable, Tuple
from fea_transfer import poolfeat, upfeat

TIME_WINDOW = 24
PRED_LEN = 6

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model(nn.Module):
    # 原 station_mask 中 False 的地方代表站点，现在我们需要 True 的地方代表站点，所以取反（~）
    station_mask: np.ndarray = ~np.load("/home/guoshuai/coding/PRD-MSGNN/final_dataset/prd_station_mask.npy")
    in_resizer = Resize(size=(128, 128))
    out_resizer = Resize(size=(163, 137))

    def __init__(self, gnn_h, cnn_h, rnn_h, rnn_l, group_em, pm25_em, gnn_layer, input_fea_num, edge_h, group_num, device) -> None:
        super(Model, self).__init__()
        self.rnn_l = rnn_l
        self.rnn_h = rnn_h
        self.gnn_layer = gnn_layer
        self.group_num = group_num
        self.device = device
        # self.superpixel_size = superpixel_size
        # self.group_num_w = group_num_w
        self.aod_encoder = CNNEncoder(7)
        self.wea_encoder = CNNEncoder(11)
        self.land_encoder = CNNEncoder(28)
        self.aod_tans = Pattern_trans()
        self.wea_tans = Pattern_trans()
        self.land_tans = Pattern_trans()
        self.cnn_predict = CNNPredict()
        # self.group_gnn: Callable[..., Tensor] = GroupGNN_Layer(group_em, 1, gnn_h)
        self.aod_group_gnn = nn.ModuleList([GroupGNN_Layer(group_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.aod_group_gnn.append(GroupGNN_Layer(gnn_h, edge_h, gnn_h))
        
        self.wea_group_gnn = nn.ModuleList([GroupGNN_Layer(group_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.wea_group_gnn.append(GroupGNN_Layer(gnn_h, edge_h, gnn_h))
        
        self.land_group_gnn = nn.ModuleList([GroupGNN_Layer(group_em, edge_h, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.land_group_gnn.append(GroupGNN_Layer(gnn_h, edge_h, gnn_h))
        # self.lstm_encoder = LSTMEncoder(
        #     input_size=gnn_h + cnn_h, hidden_size=rnn_h, num_layers=rnn_l
        # )
        # self.lstm_decoder: Callable[..., Tuple[Tensor, Tensor, Tensor]] = LSTMDecoder(
        #     input_size=pm25_em, hidden_size=rnn_h, num_layers=rnn_l
        # )
        self.aod_superpixel_encode = SuperPixel_encoder(1, 8, device, self.group_num)
        self.wea_superpixel_encode = SuperPixel_encoder(5, 8, device, self.group_num)
        self.land_superpixel_encode = SuperPixel_encoder(22, 8, device, self.group_num)
        self.aod_group_embed: Callable[..., Tensor] = nn.Sequential(
            nn.Linear(7, group_em), nn.ReLU()
        )
        self.land_group_embed: Callable[..., Tensor] = nn.Sequential(
            nn.Linear(28, group_em), nn.ReLU()
        )
        self.wea_group_embed: Callable[..., Tensor] = nn.Sequential(
            nn.Linear(11, group_em), nn.ReLU()
        )
        self.aod_edge_inf = nn.Sequential(
            nn.Linear(group_em * 2, edge_h), nn.ReLU(inplace=True)
        )
        self.wea_edge_inf = nn.Sequential(
            nn.Linear(group_em * 2, edge_h), nn.ReLU(inplace=True)
        )
        self.land_edge_inf = nn.Sequential(
            nn.Linear(group_em * 2, edge_h), nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )
        

    @staticmethod
    def batch_input(x: Tensor, edge_w: Tensor, edge_conn: Tensor):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        edge_index = torch.clone(edge_conn)
        for i in range(edge_conn.size(0)):
            edge_index[i, :] = torch.add(edge_index[i, :], i * sta_num)
        edge_index = edge_index.transpose(0, 1)
        edge_index = edge_index.reshape(2, -1)
        return x, edge_w, edge_index
    
    # def edge_encode(self, g_x:Tensor, g_edge_index):
    #     g_edge_index = g_edge_index.clone().transpose(0, 1)
    #     row, col = g_edge_index
    #     g_x_concat = torch.concat((g_x[:, row], g_x[:, col]), dim=2)
    #     g_edge_w = self.edge_inf(g_x_concat)
    #     g_edge_index = g_edge_index.unsqueeze(dim=0)
    #     g_edge_index = g_edge_index.repeat_interleave(g_x.shape[0], dim=0)
    #     # g_edge_index = g_edge_index.transpose(1, 2)
    #     return g_edge_index, g_edge_w
    
    @staticmethod
    def edge_encode(g_x:Tensor, g_edge_index, edge_inf):
        g_edge_index = g_edge_index.clone().transpose(0, 1)
        row, col = g_edge_index
        g_x_concat = torch.concat((g_x[:, row], g_x[:, col]), dim=2)
        g_edge_w = edge_inf(g_x_concat)
        g_edge_index = g_edge_index.unsqueeze(dim=0)
        g_edge_index = g_edge_index.repeat_interleave(g_x.shape[0], dim=0)
        # g_edge_index = g_edge_index.transpose(1, 2)
        return g_edge_index, g_edge_w

        
    def score_com(self, new_x_aod):
        # shape:[32, 3*32, 128, 128]
        new_x = new_x_aod.clone()
        new_x = new_x.permute(0, 2, 3, 1)
        atte_score: Tensor = self.attention(new_x)

        # 形状变换: [32, 128, 128, 1]->[32, 1 128, 128]
        atte_score = atte_score.permute(0, 3, 1, 2)
        

        
        return atte_score

    def forward(
        self,
        x: Tensor,
        land_use,
        device,
    ):
        # 转换输入 x 的shape：[32,156,151,7]->[32,7,156,151]
        x = x.permute(0, 3, 1, 2)
        air = x[:, :6, :, :]

        ''' 取出 aod 值，计算 assi_matrix，将空气质量比较类似的 pixel 聚合到一起'''
        # aod 的 shape：[32,156,151,1], aod_assi_matrix shape:[32, 156, 151, 16, 16]
        aod = x[:, 6, :, :].unsqueeze(1)

        # 计算在 AOD superpixel 下的 group
        # 先将原始输入作一下 resize，[32,7,156,151]->[32,7,128,128]
        aod_x = torch.cat([air, aod], dim=1)

        # 计算下层特征
        aod_x_spatial_fea = self.aod_encoder(aod_x)

        # 计算 aod_assi_matrix，aod_assi_matrix:[b, 256, 128, 128]
        aod_assi_matrix = self.aod_superpixel_encode(aod_x_spatial_fea, device)
        
        resize_aodx = self.in_resizer(aod_x)

        # 更新上层特征
        # 新传变换:[32, 256, 128, 128]->[32, 128, 128, 256]
        aod_assi_matrix = aod_assi_matrix.permute(0, 2, 3, 1)
        g_x_aod = torch.einsum('abcd,acde->abe', resize_aodx, aod_assi_matrix)
        # g_x_aod = poolfeat(resize_aodx, aod_assi_matrix, device, self.superpixel_size, self.superpixel_size)

        # 形状转换 [32, 7, 256]->[32, 256, 7]
        g_x_aod = g_x_aod.permute(0, 2, 1)

        # 计算每个 superpixel 的特征表示
        g_x_aod = self.aod_group_embed(g_x_aod)

        # edge_index shape:[batch_size, edge_num, 2]->[batch_size, 2, edge_num]
        g_edge_index = torch.tensor(
            [
                [i, j]
                for i in range(self.group_num)
                for j in range(self.group_num)
                if i != j
            ],
            dtype=int,
            device=self.device,
        )
        # 建模 superpixel 之间的关系
        g_edge_index_aod, g_edge_w_aod = self.edge_encode(g_x_aod, g_edge_index, self.aod_edge_inf)

        # 输入 batchinput 将数据处理成适合 GNN 的输入，并输入 GNN 网络
        g_x_aod, g_edge_w_aod, g_edge_index_aod = self.batch_input(g_x_aod, g_edge_w_aod, g_edge_index_aod)
        for i in range(self.gnn_layer):
            g_x_aod = self.aod_group_gnn[i](g_x_aod, g_edge_index_aod, g_edge_w_aod)
        # shape [batch*256, 32]->[32, 256, 32]->[32(b), 32, 16, 16]
        gx_spatial_fea_aod = g_x_aod
        gx_spatial_fea_aod = gx_spatial_fea_aod.reshape(
            -1, self.group_num, gx_spatial_fea_aod.shape[-1]
        )

        # new_x_aod = upfeat(gx_spatial_fea_aod, aod_assi_matrix, self.superpixel_size, self.superpixel_size)
        new_x_aod = torch.einsum('abcd,ade->abce', aod_assi_matrix, gx_spatial_fea_aod)

        # 形状变换
        new_x_aod = new_x_aod.permute(0, 3, 1, 2)

        # 此时 new_x shape：[32, 32, 128, 128]
        # 形状变化:[32, 32, 128, 128]->[32, 32, 156, 151]
        new_x_aod = self.out_resizer(new_x_aod)

        ''' 取出 weather 值，计算 assi_matrix, 将天气情况比较类似的 pixel 聚合到一起'''
        # wea 的 shape：[32,156,151,5], aod_assi_matrix shape:[32, 156, 151, 16, 16]
        weather = x[:, 7:12, :, :]

        # 先将原始输入作一下 resize，[32,7,156,151]->[32,7,128,128]
        wea_x = torch.cat([air, weather], dim=1)

        # 计算下层特征
        wea_x_spatial_fea = self.wea_encoder(wea_x)

        # 计算 wea_assi_matrix，wea_assi_matrix:[b, 9, 128, 128]
        wea_assi_matrix = self.wea_superpixel_encode(wea_x_spatial_fea, device)

        resize_weax = self.in_resizer(wea_x)

        # 计算在 AOD superpixel 下的 group
        wea_assi_matrix = wea_assi_matrix.permute(0, 2, 3, 1)
        g_x_wea = torch.einsum('abcd,acde->abe', resize_weax, wea_assi_matrix)
        
        # g_x_wea = poolfeat(resize_weax, wea_assi_matrix, device, self.superpixel_size, self.superpixel_size)

        # 形状转换 [32, 7, 256]->[32, 256, 7]
        g_x_wea = g_x_wea.permute(0, 2, 1)

        # 计算每个 superpixel 的特征表示
        g_x_wea = self.wea_group_embed(g_x_wea)

        # edge_index shape:[batch_size, edge_num, 2]->[batch_size, 2, edge_num]
        # 建模 superpixel 之间的关系
        g_edge_index_wea, g_edge_w_wea = self.edge_encode(g_x_wea, g_edge_index, self.wea_edge_inf)

        # 输入 batchinput 将数据处理成适合 GNN 的输入，并输入 GNN 网络
        g_x_wea, g_edge_w_wea, g_edge_index_wea = self.batch_input(g_x_wea, g_edge_w_wea, g_edge_index_wea)
        for i in range(self.gnn_layer):
            g_x_wea = self.wea_group_gnn[i](g_x_wea, g_edge_index_wea, g_edge_w_wea)
        # shape [batch*256, 32]->[32, 16, 16, 32]->[32(b), 32, 16, 16]
        gx_spatial_fea_wea = g_x_wea
        gx_spatial_fea_wea = gx_spatial_fea_wea.reshape(
            -1, self.group_num, gx_spatial_fea_wea.shape[-1]
        )

        new_x_wea = torch.einsum('abcd,ade->abce', wea_assi_matrix, gx_spatial_fea_wea)

        # 形状变换
        new_x_wea = new_x_wea.permute(0, 3, 1, 2)

        # 此时 new_x shape：[32, 32, 128, 128]
        # 形状变化:[32, 32, 128, 128]->[32, 32, 156, 151]
        new_x_wea = self.out_resizer(new_x_wea)

        ''' 取出 land_assi_matrix, 将 land_use 比较类似的 pixel 聚合到一起'''
        # land 的 shape：[32,156,151,22], aod_assi_matrix shape:[32, 156, 151, 16, 16]
        # 形状变化
        land_use = land_use.unsqueeze(dim=0)
        land_use = land_use.permute(0, 3, 1, 2)
        # 先将原始输入作一下 resize，[32,7,156,151]->[32,7,128,128]
        land_use = land_use.repeat_interleave(x.shape[0], dim=0)
        land_x = torch.cat([air, land_use], dim=1)

        # 计算下层特征
        land_x_spatial_fea = self.land_encoder(land_x)

        # 计算 wea_assi_matrix，wea_assi_matrix:[b, 9, 128, 128]
        land_assi_matrix = self.land_superpixel_encode(land_x_spatial_fea, device)

        resize_landx = self.in_resizer(land_x)
        
        # 更新上层特征
        # 新传变换:[32, 256, 128, 128]->[32, 128, 128, 256]
        land_assi_matrix = land_assi_matrix.permute(0, 2, 3, 1)
        g_x_land = torch.einsum('abcd,acde->abe', resize_landx, land_assi_matrix)

        # 形状转换 [32, 7, 256]->[32, 256, 7]
        g_x_land = g_x_land.permute(0, 2, 1)
        
        # 计算每个 superpixel 的特征表示
        g_x_land = self.land_group_embed(g_x_land)

        # edge_index shape:[batch_size, edge_num, 2]->[batch_size, 2, edge_num]
        # 建模 superpixel 之间的关系
        g_edge_index_land, g_edge_w_land = self.edge_encode(g_x_land, g_edge_index, self.land_edge_inf)

        # 输入 batchinput 将数据处理成适合 GNN 的输入，并输入 GNN 网络
        g_x_land, g_edge_w_land, g_edge_index_land = self.batch_input(g_x_land, g_edge_w_land, g_edge_index_land)
        for i in range(self.gnn_layer):
            g_x_land = self.land_group_gnn[i](g_x_land, g_edge_index_land, g_edge_w_land)
        # shape [batch*256, 32]->[32, 16, 16, 32]->[32(b), 32, 16, 16]
        gx_spatial_fea_land = g_x_land
        gx_spatial_fea_land = gx_spatial_fea_land.reshape(
            -1, self.group_num, gx_spatial_fea_land.shape[-1]
        )

        # new_x_aod = upfeat(gx_spatial_fea_aod, aod_assi_matrix, self.superpixel_size, self.superpixel_size)
        new_x_land = torch.einsum('abcd,ade->abce', land_assi_matrix, gx_spatial_fea_land)

        # 形状变换
        new_x_land = new_x_land.permute(0, 3, 1, 2)

        # 此时 new_x shape：[32, 32, 128, 128]
        # 形状变化:[32, 32, 128, 128]->[32, 32, 156, 151]
        new_x_land = self.out_resizer(new_x_land)

        # 形状变换
        aod_x_spatial_fea = self.out_resizer(aod_x_spatial_fea)
        wea_x_spatial_fea = self.out_resizer(wea_x_spatial_fea)
        land_x_spatial_fea = self.out_resizer(land_x_spatial_fea)
        
        # 此时 x_spatial_fea 的形状是：[32, 16, 156, 151]

        aod_group_cat = torch.cat([aod_x_spatial_fea, new_x_aod], dim=1)
        wea_group_cat = torch.cat([wea_x_spatial_fea, new_x_wea], dim=1)
        land_group_cat = torch.cat([land_x_spatial_fea, new_x_land], dim=1)

        # 形状变换一下，都给变到 32
        aod_group_trans =  self.aod_tans(aod_group_cat)
        wea_group_trans =  self.wea_tans(wea_group_cat)
        land_group_trans =  self.land_tans(land_group_cat)

        # 计算重要性
        aod_impor = self.score_com(aod_group_trans)
        wea_impor = self.score_com(wea_group_trans)
        land_impor = self.score_com(land_group_trans)

        # x_group_cat:[32, 32+32+32, 156, 151 ]
        x_group_cat = torch.cat([aod_group_trans * aod_impor, wea_group_trans * wea_impor, land_group_trans * land_impor], dim=1)

        # 输入 CNN 当中做 estimate
        # predict shape:[32, 1, 156, 151]
        outputs = self.cnn_predict(x_group_cat) # 改
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3])
        # end = time.perf_counter_ns()
        # print(f"Model.forward() time: {(end - start) / 1000000 :,} ms")
        return outputs, aod_assi_matrix, wea_assi_matrix, land_assi_matrix

class SuperPixel_encoder(nn.Module):
    in_resizer = Resize(size=(128, 128))
    out_resizer = Resize(size=(163, 137))

    def __init__(self, in_channel, out_channel, device, group_num, batch_norm=True) -> None:
        super().__init__()
        self.assign_ch = group_num
        self.conv1 = self._conv(batch_norm, 16, 32)
        self.conv2 = self._conv(batch_norm, 32, 64)
        self.pred_mask0 = self.predict_mask(64, self.assign_ch)


        print(f"[SuperPixel_encoder] Number of params: {_count_parameters(self):,}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    # def expand_sp(self, sp_feat):
    #     sp_feat = nn.functional.pad(sp_feat, [1,1,1,1])
    #     b, c, h, w = sp_feat.shape
    #     output_list = []
    #     #this loop is acceptable due to the lower h, w
    #     for i in range(1,h-1):
    #         row_list = []
    #         for j in range(1,w-1):
    #             sp_patch = sp_feat[:,:, (i-1):(i+2), (j-1):(j+2)]
    #             sp_patch = sp_patch.repeat(1,1,8,8)
    #             row_list.append(sp_patch)

    #         output_list.append(torch.cat(row_list, dim=-1))
        
    #     output = torch.cat(output_list, dim=-2)

    #     return output

    # def expand_pixel(self, pixel_feat):
    #     b,c,h,w = pixel_feat.shape
    #     pixel_feat = pixel_feat.view(b,c,h,1,w,1)
    #     pixel_feat = pixel_feat.repeat(1,1,1,3,1,3)
    #     pixel_feat = pixel_feat.reshape(b,c,h*3,w*3)
        
    #     pixel_feat = pixel_feat * self.mask_select
    
    #     return pixel_feat
    
    def forward(self, x: Tensor, device):
        # 计算下采样所用时间
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        mask0 = self.pred_mask0(conv2)
        prob0 = nn.functional.softmax(mask0, dim=1)

        # 计算上采样所用时间
        
        # print(f'上采样所用时间：{up_consume}')

        return prob0

    @staticmethod
    def predict_mask(in_planes, channel=9):
        return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)
    
    @staticmethod
    def _conv(
        batch_norm, in_planes, out_planes, kernel_size=3, stride=1
    ) -> Callable[..., Tensor]:
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=True,
                ),
                nn.LeakyReLU(0.1),
            )

    @staticmethod
    def _deconv(in_planes, out_planes) -> Callable[..., Tensor]:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.LeakyReLU(0.1),
        )


class CNNPredict(nn.Module):
    # in_resizer = Resize(size=(128, 128))
    # out_resizer = Resize(size=(156, 151))

    def __init__(self, batch_norm=True) -> None:
        super().__init__()
        self._conv0a = self._conv(batch_norm, 96, 64)
        self._conv1a = self._conv(batch_norm, 64, 32)
        self.predict = self._conv(batch_norm, 32, 1)
        

        print(f"[CNNPredict] Number of params: {_count_parameters(self):,}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x: Tensor):
        # x = self.in_resizer(x)
        out0 = self._conv0a(x)  # 5*5
        out1 = self._conv1a(out0)
        out = self.predict(out1)  # 11*11
        return out

    @staticmethod
    def _conv(
        batch_norm, in_planes, out_planes, kernel_size=3, stride=1
    ) -> Callable[..., Tensor]:
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=True,
                ),
                nn.LeakyReLU(0.1),
            )

class Pattern_trans(nn.Module):
    # in_resizer = Resize(size=(128, 128))
    # out_resizer = Resize(size=(156, 151))

    def __init__(self, batch_norm=True) -> None:
        super().__init__()
        self._conv0a = self._conv(batch_norm, 48, 32)
        
        print(f"[Pattern_trans] Number of params: {_count_parameters(self):,}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x: Tensor):
        # x = self.in_resizer(x)
        out0 = self._conv0a(x)  # 5*5

        return out0

    @staticmethod
    def _conv(
        batch_norm, in_planes, out_planes, kernel_size=1, stride=1
    ) -> Callable[..., Tensor]:
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=True,
                ),
                nn.LeakyReLU(0.1),
            )
        

class CNNEncoder(nn.Module):
    in_resizer = Resize(size=(128, 128))
    out_resizer = Resize(size=(163, 137))

    def __init__(self, in_channel = 0, batch_norm=True) -> None:
        super().__init__()
        self._conv0a = self._conv(batch_norm, in_channel, 16)
        self._conv0b = self._conv(batch_norm, 16, 16)
        self._drop0 = nn.Dropout(0.2)

        self._conv1a = self._conv(batch_norm, 16, 32, stride=2)
        self._conv1b = self._conv(batch_norm, 32, 32)
        self._drop1 = nn.Dropout(0.4)

        self._conv2a = self._conv(batch_norm, 32, 64, stride=2)
        self._conv2b = self._conv(batch_norm, 64, 64)
        self._drop2 = nn.Dropout(0.5)

        self._conv3a = self._conv(batch_norm, 64, 128, stride=2)
        self._conv3b = self._conv(batch_norm, 128, 128)

        # self._conv4a = self._conv(batch_norm, 128, 256, stride=2)
        # self._conv4b = self._conv(batch_norm, 256, 256)

        # self._deconv3 = self._deconv(256, 128)
        # self._conv3_1 = self._conv(batch_norm, 256, 128)

        self._deconv2 = self._deconv(128, 64)
        self._conv2_1 = self._conv(batch_norm, 128, 64)

        self._deconv1 = self._deconv(64, 32)
        self._conv1_1 = self._conv(batch_norm, 64, 32)

        self._deconv0 = self._deconv(32, 16)
        self._conv0_1 = self._conv(batch_norm, 32, 16)

        print(f"[CNNEncoder] Number of params: {_count_parameters(self):,}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x: Tensor):
        x = self.in_resizer(x)
        out1 = self._conv0b(self._conv0a(x))  # 5*5
        out1 = self._drop0(out1)
        out2 = self._conv1b(self._conv1a(out1))  # 11*11
        out2 = self._drop1(out2)
        out3 = self._conv2b(self._conv2a(out2))  # 23*23
        out3 = self._drop2(out3)
        out4 = self._conv3b(self._conv3a(out3))  # 47*47
        # out5 = self._conv4b(self._conv4a(out4))  # 95*95

        # out_deconv3 = self._deconv3(out5)
        # concat3 = torch.cat((out4, out_deconv3), 1)
        # out_conv3_1 = self._conv3_1(concat3)

        out_deconv2 = self._deconv2(out4)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self._conv2_1(concat2)

        out_deconv1 = self._deconv1(out_conv2_1)
        # out_deconv1 = self._drop2(out_deconv1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self._conv1_1(concat1)
        # out_conv1_1 = self._drop2(out_conv1_1)

        out_deconv0 = self._deconv0(out_conv1_1)
        # out_deconv0 = self._drop2(out_deconv0)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self._conv0_1(concat0)
        # out_conv0_1 = self._drop1(out_conv0_1)

        # out_resized = self.out_resizer(out_conv0_1)

        return out_conv0_1

    @staticmethod
    def _conv(
        batch_norm, in_planes, out_planes, kernel_size=3, stride=1
    ) -> Callable[..., Tensor]:
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=True,
                ),
                nn.LeakyReLU(0.1),
            )

    @staticmethod
    def _deconv(in_planes, out_planes) -> Callable[..., Tensor]:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.LeakyReLU(0.1),
        )


class GroupGNN_Layer(nn.Module):
    def __init__(self, node_h, edge_h, gnn_h) -> None:
        super(GroupGNN_Layer, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(2 * node_h + edge_h, gnn_h), nn.ReLU(inplace=True)
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(node_h + gnn_h, gnn_h), nn.ReLU(inplace=True)
        )

        print(f"[GroupGNN] Number of params: {_count_parameters(self):,}")

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

