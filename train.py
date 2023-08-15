from pathlib import Path
from dataset import Dataset
from model_add_all_new_super_all_learn_input_crop_atte import Model
from superpixel_loss import compute_superpixel_loss
import torch.utils.data as Data
import argparse
import time
import torch.nn as nn
import torch
import numpy as np
import pickle
import math
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# 构建传入的参数
parser = argparse.ArgumentParser(description="AQ forecasting with AOD")
parser.add_argument("--device", type=str, default="cuda:1", help="")
parser.add_argument("--run_times", type=int, default=1, help="") #代码运行多少次
parser.add_argument("--epoch", type=int, default=300, help="")
parser.add_argument("--batch_size", type=int, default=48, help="")
parser.add_argument("--w_rate", type=int, default=50, help="")
parser.add_argument("--group_num", type=int, default=14, help="")
parser.add_argument("--gnn_h", type=int, default=32, help="")
parser.add_argument("--cnn_h", type=int, default=16, help="")
parser.add_argument("--rnn_h", type=int, default=64, help="")
parser.add_argument("--rnn_l", type=int, default=1, help="")
parser.add_argument("--group_em", type=int, default=16, help="group embedding")
parser.add_argument("--pm25_em", type=int, default=16, help="pm25 embedding")
parser.add_argument("--gnn_layer", type=int, default=2, help="gnn layer num")
parser.add_argument("--input_fea_num", type=int, default=14, help="input feature num")
parser.add_argument("--lr", type=float, default=0.001, help="lr")
parser.add_argument("--wd", type=float, default=0.001, help="weight decay")
parser.add_argument("--edge_h", type=int, default=12, help="edge h")
parser.add_argument("--mask", type=str, default="prd_model_g_14", help="mask model")
parser.add_argument("--only_evaluate", action="store_true", help="是否只推理？")
parser.add_argument("--station_cell_num", type=int, default=112, help="有多少个 cell 是有 station 的")
parser.add_argument("--mask_rate", type=float, default=0.2, help="mask 掉多少 cell")
parser.add_argument("--aod_sem_weight", type=float, default=15, help="aod superpixel 分割 sem 的重要性")
parser.add_argument("--wea_sem_weight", type=float, default=0.0001, help="wea superpixel 分割 sem 的重要性")
parser.add_argument("--land_sem_weight", type=float, default=0.06, help="land superpixel 分割 sem 的重要性")
parser.add_argument("--aod_super_weight", type=float, default= 0.7, help="aod superpixel 的重要性")
parser.add_argument("--wea_super_weight", type=float, default=0.3, help="wea superpixel 的重要性")
parser.add_argument("--land_super_weight", type=float, default=0.5, help="land superpixel 的重要性")
parser.add_argument("--superpixel_loss_weight", type=float, default=0.5, help="superpixel loss 重要性")
args = parser.parse_args()


# 创建被 mask 和被保留的站点
mask_station = []
remain_station = []

step = 1 / args.mask_rate

for i in range(args.station_cell_num):
    if i % step == 0:
        mask_station.append(i)
    else:
        remain_station.append(i)

# 读取出 station 的位置
with open('/home/guoshuai/coding/PRD-MSGNN/final_dataset/prd_station_list.pkl', 'rb') as f:
    station_list = pickle.load(f)

# 创建 dataset 和 dataloader, 此时是 unshuffle 的
indices_dir = Path("/home/guoshuai/coding/AQ_withAOD/final_dataset")
train_dataset = Dataset(indices_dir / "unshuffled_train_indices.npy", mask_station, remain_station)
valid_dataset = Dataset(indices_dir / "unshuffled_valid_indices.npy", mask_station, remain_station)
test_dataset = Dataset(indices_dir / "unshuffled_test_indices.npy", mask_station, remain_station)

print("train_dataset length:", len(train_dataset))
print("valid_dataset length:", len(valid_dataset))
print("test_dataset length:", len(test_dataset))

train_loader = Data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)
val_loader = Data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)
test_loader = Data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

device = args.device

# 整个模型总共训练 run_times 次
for _ in range(args.run_times):
    start = time.time()
    model = Model(
        args.gnn_h,
        args.cnn_h,
        args.rnn_h,
        args.rnn_l,
        args.group_em,
        args.pm25_em,
        args.gnn_layer,
        args.input_fea_num,
        args.edge_h,
        args.group_num,
        args.device
    ).to(device)

    # 加载 station_mask
    station_mask = ~np.load("/home/guoshuai/coding/PRD-MSGNN/final_dataset/prd_station_mask.npy")
    station_mask = torch.tensor(station_mask, device=device)
    # 加载 land_use
    geo_spatial = np.load("/home/guoshuai/coding/PRD-MSGNN/final_dataset/prd_landuse.npy")
    land_use = geo_spatial
    land_use = land_use.astype(np.float32)
    land_use = torch.tensor(land_use, device=device)
    
    if not args.only_evaluate:
        in_resizer = Resize(size=(128, 128))
        model_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model pram num:", model_num)
        criterion = nn.L1Loss(reduction="sum")
        all_params = model.parameters()
        
        optimizer = torch.optim.Adam(
            [{"params": all_params}],
            lr=args.lr,
            weight_decay=args.wd,
        )

        val_loss_min = np.inf
        global_steps = 0
        for epoch in range(args.epoch):
            for i, data in enumerate(train_loader):
                forward_start = time.time()
                data = [item.to(device, non_blocking=False) for item in data]
                x, y_full = data

                # 将 y_full mask 成只有 station 的地方,改
                y = y_full[:, station_mask]
                
                # 将数据输入模型
                # aod_assi_matrix_soft shape:[b, 9, 128, 128]
                outputs, aod_assi_matrix_soft, wea_assi_matrix_soft, land_assi_matrix_soft = model(x, land_use, device)

                # 计算前向所用时间
                forward_end = time.time()
                forward_consume = forward_end - forward_start
                # print(f'前向计算时间：{forward_consume}')

                # 计算反向所用时间
                backward_start = time.time()

                # superpixel loss
                aod = x[:, :, :, 6].unsqueeze(-1)
                wea = x[:, :, :, 7:12]
                lon_lat = x[:, :, :, -2:]
                # aod_with_geo shape:[b, 156, 151, 3]
                aod_with_geo = torch.cat((aod, lon_lat), dim=-1)
                wea_with_geo = torch.cat((wea, lon_lat), dim=-1)
                land_with_geo = torch.cat((land_use, lon_lat[0, :, :, :]), dim=-1)
                land_with_geo = land_with_geo.unsqueeze(dim=0)
                land_with_geo = land_with_geo.repeat_interleave(args.batch_size, dim=0)

                # shape 变化:[b, h, w, 3]->[b, 3, h, w]->[b, 3, 128, 128]
                aod_with_geo = aod_with_geo.permute(0, 3, 1, 2)
                aod_with_geo = in_resizer(aod_with_geo)
                wea_with_geo = wea_with_geo.permute(0, 3, 1, 2)
                wea_with_geo = in_resizer(wea_with_geo)
                land_with_geo = land_with_geo.permute(0, 3, 1, 2)
                land_with_geo = in_resizer(land_with_geo)

                aod_superpixel_loss, aod_sem_loss = compute_superpixel_loss(aod_assi_matrix_soft, aod_with_geo, args.aod_sem_weight)
                wea_superpixel_loss, wea_sem_loss = compute_superpixel_loss(wea_assi_matrix_soft, wea_with_geo, args.wea_sem_weight)
                land_superpixel_loss, land_sem_loss = compute_superpixel_loss(land_assi_matrix_soft, land_with_geo, args.land_sem_weight)


                # estimate loss mask
                outputs = outputs[:, station_mask]

                # total loss
                estimate_loss = criterion(y, outputs)
                loss = (1 - args.superpixel_loss_weight) * estimate_loss + args.superpixel_loss_weight * (args.aod_super_weight * aod_superpixel_loss + args.wea_super_weight * wea_superpixel_loss + args.land_super_weight * land_superpixel_loss)
                writer.add_scalar('Loss/train', loss, global_steps)
                model.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算反向所有时间
                backward_end = time.time()
                backward_consume = backward_end - backward_start
                # print(f'反向计算所用时间：{backward_consume}')

                if epoch % 10 == 0 and i % 20 == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, estimate_loss: {:.4f}, aod_superpixel_loss: {:.4f}, wea_superpixel_loss: {:.4f}, land_superpixel_loss: {:.4f}".format(
                            epoch,
                            args.epoch,
                            i,
                            int(len(train_dataset) / args.batch_size),
                            loss.item(),
                            estimate_loss.item(),
                            aod_superpixel_loss.item(),
                            wea_superpixel_loss.item(),
                            land_superpixel_loss.item()
                        )
                    )
                global_steps += 1

            if epoch % 5 == 0:
                print("Validating...")
                with torch.no_grad():
                    val_loss = 0
                    for j, data_val in enumerate(val_loader):
                        data_val = [item.to(device, non_blocking=True) for item in data_val]
                        x_val, y_val_full = data_val

                        # 将有站点的 mask 出来，改
                        y_val = y_val_full[:, station_mask]
                        
                        outputs_val, aod_assi_matrix_soft_val, wea_assi_matrix_soft_val, land_assi_matrix_soft_val = model(
                            x_val, land_use, device
                        )

                        # superpixel loss
                        aod_val = x_val[:, :, :, 6].unsqueeze(-1)
                        wea_val = x_val[:, :, :, 7:12]
                        lon_lat_val = x_val[:, :, :, -2:]
                        # aod_with_geo shape:[b, 156, 151, 3]
                        aod_with_geo_val = torch.cat((aod_val, lon_lat_val), dim=-1)
                        wea_with_geo_val = torch.cat((wea_val, lon_lat_val), dim=-1)
                        land_with_geo_val = torch.cat((land_use, lon_lat_val[0, :, :, :]), dim=-1)
                        land_with_geo_val = land_with_geo_val.unsqueeze(dim=0)
                        land_with_geo_val = land_with_geo_val.repeat_interleave(args.batch_size, dim=0)

                        # shape 变化:[b, h, w, 3]->[b, 3, h, w]->[b, 3, 128, 128]
                        aod_with_geo_val = aod_with_geo_val.permute(0, 3, 1, 2)
                        aod_with_geo_val = in_resizer(aod_with_geo_val)
                        wea_with_geo_val = wea_with_geo_val.permute(0, 3, 1, 2)
                        wea_with_geo_val = in_resizer(wea_with_geo_val)
                        land_with_geo_val = land_with_geo_val.permute(0, 3, 1, 2)
                        land_with_geo_val = in_resizer(land_with_geo_val)

                        aod_superpixel_loss_val, aod_sem_loss_val = compute_superpixel_loss(aod_assi_matrix_soft_val, aod_with_geo_val, args.aod_sem_weight)
                        wea_superpixel_loss_val, wea_sem_loss_val = compute_superpixel_loss(wea_assi_matrix_soft_val, wea_with_geo_val, args.wea_sem_weight)
                        land_superpixel_loss_val, land_sem_loss_val = compute_superpixel_loss(land_assi_matrix_soft_val, land_with_geo_val, args.land_sem_weight)

                        outputs_val = outputs_val[:, station_mask]
                        # batch_loss = criterion(y_val, outputs_val)
                        batch_loss = (1 - args.superpixel_loss_weight) * criterion(y_val, outputs_val) + args.superpixel_loss_weight * (args.aod_super_weight * aod_superpixel_loss_val + args.wea_super_weight * wea_superpixel_loss_val + args.land_super_weight * land_superpixel_loss_val)
                        val_loss += batch_loss.item()

                    print("Epoch:", epoch, ", val_loss:", val_loss)
                    if val_loss < val_loss_min:
                        torch.save(
                            model.state_dict(),
                            "./save_model/model_para_" + args.mask + ".ckpt",
                        )
                        val_loss_min = val_loss
                        print("parameters have been updated during epoch ", epoch)
    
    # 测试部分
    mae_loss = 0
    rmse_loss = 0
    count = 0

    def cal_loss(outputs, y):
        global mae_loss, rmse_loss, count
        for i in mask_station:
            a = station_list[i][0]
            b = station_list[i][1]
            gt = y[:, a, b]
            out = outputs[:, a, b]
            temp_loss = torch.abs(gt - out)
            mae_loss = mae_loss + torch.sum(temp_loss)
            temp_loss = torch.pow(temp_loss, 2)
            rmse_loss = rmse_loss + torch.sum(temp_loss)
            count = count + args.batch_size

        
    with torch.no_grad():
        model.load_state_dict(
            torch.load("./save_model/model_para_" + args.mask +  ".ckpt")
        )

        for i, data in enumerate(test_loader):
            data = [item.to(device, non_blocking=True) for item in data]
            x, y_full = data

            # 将有 station 的 cell mask 出来
            # y = y_full[:, station_mask, :]
            

            outputs, aod_assi_matrix_soft, wea_assi_matrix_soft, land_assi_matrix_soft = model(x, land_use, device)
            # outputs = outputs[:, station_mask]
            cal_loss(outputs, y_full)

        mae_loss = mae_loss / count
        rmse_loss = math.sqrt(rmse_loss / count)
        # mae_loss = mae_loss.mean(dim=0)
        # rmse_loss = rmse_loss.mean(dim=0)

        end = time.time()
        print("Running time: %s Seconds" % (end - start))

        mae_loss = mae_loss.cpu()
        # rmse_loss = rmse_loss.cpu()

        print("mae:", mae_loss)
        print("rmse:", rmse_loss)
        print("count:", count)
