"""
python transfomer_train.py --data_root ./datasets/Data_MP_XMnO/cifs_xmno --num_workers 4 --batch_size 32 --steps_per_epoch 800

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from DeepTransform import DeepTransform
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
import numpy as np
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from ema import EMAHelper
from loss_function import DistanceL1Loss
from graph_utils import get_edge_dist_relaxed, get_edge_dist_displace
from utils import *
import warnings
warnings.filterwarnings("ignore")


# reorders the atoms so that atom 0 has the lowest z value, then ascends from there.

def reorder_transform(data):
    pos_tensors = []
    n_atoms = data.natoms

    for i, pos in enumerate(data.pos_u):
        pos_tensors.append((pos, i))
    
    
    pos_tensors.sort(key=lambda item: item[0][0])
    pos_tensors.sort(key=lambda item: item[0][1])
    pos_tensors.sort(key=lambda item: item[0][2])

    # swap x, pos_r, pos_u (done already),




    x_list = [0] * n_atoms
    pos_r_list = [0] * n_atoms

    for i in range(n_atoms):
        x_list[i] = data.x[pos_tensors[i][1]]
        pos_r_list[i] = data.pos_r[pos_tensors[i][1]]
    

    sources = [pos_tensors[i][1] for i in data.edge_index[0]]
    dests = [pos_tensors[i][1] for i in data.edge_index[1]]

    num_edges = len(sources)
    
    edge_tuples = []
    for i in range(num_edges):
        edge_tuples.append((sources[i], dests[i], i))

    # sort by one (source)

    edges_in_cell = n_atoms * (n_atoms - 1)

    cell_edges = edge_tuples[0:edges_in_cell]
    intercell_edges = edge_tuples[edges_in_cell:]

    # sort by sources first
    cell_edges.sort(key = lambda item: item[0])
    intercell_edges.sort(key = lambda item: item[0])

    # then sort by the other (dest)
    cell_edges.sort(key = lambda item: item[1])
    intercell_edges.sort(key = lambda item: item[1])   

    edge_tuples = []
    edge_tuples.extend(cell_edges)
    edge_tuples.extend(intercell_edges)

    offset_list = []

    for i in range(num_edges):
        offset_list.append(data.cell_offsets[edge_tuples[i][2]])
        
    # checking
    """
    for i in range(n_atoms):
        actual_x = x_list[i].tolist()
        actual_pos = pos_r_list[i].tolist()

        expected_index = pos_tensors[i][1]
        expected_x = data.x[expected_index].tolist()
        expected_pos = data.pos_r[expected_index].tolist()

        assert(actual_x == expected_x)
        assert(actual_pos == expected_pos)

    
    # testing the transfer of edge index and attributes
    expected = []
    actual = []
    for i in range(len(edge_tuples)):
        expected_index = edge_tuples[i][2]
        expected_tuple = (pos_tensors[data.edge_index[0][expected_index].item()][1], pos_tensors[data.edge_index[1][expected_index].item()][1], data.cell_offsets[expected_index].tolist())
        actual_tuple = (edge_tuples[i][0], edge_tuples[i][1], offset_list[i].tolist())


        assert(expected_tuple == actual_tuple)
       """ 
    data.x = torch.stack(x_list)
    data.pos_u = torch.stack([item[0] for item in pos_tensors])
    data.pos_u = torch.stack(pos_r_list)
    data.edge_index = torch.stack([torch.tensor((item[0], item[1])) for item in edge_tuples]).transpose(0, 1)
    data.cell_offsets = torch.stack(offset_list)

    return (data)



# %%
def val(model, dataloader, criterion_dist, criterion_cell, device):
    model.eval()

    running_loss = AverageMeter()
    running_loss_dist_displace = AverageMeter()
    running_loss_dist_relaxed = AverageMeter()
    running_loss_cell = AverageMeter()

    pred_dist_displace_list = []
    label_dist_displace_list = []

    pred_dist_relaxed_list = []
    label_dist_relaxed_list = []

    pred_cell_list = []
    label_cell_list =[]

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred_dist_displace, pred_var_displace, pred_dist_relaxed, pred_var_relaxed, pred_cell = model(data)
            label_dist_displace, label_dist_relaxed, label_cell \
                                            = get_edge_dist_displace(data), get_edge_dist_relaxed(data), data.cell_r
        
            loss_dist_displace = criterion_dist(pred_dist_displace, pred_var_displace, label_dist_displace)
            loss_dist_relaxed = criterion_dist(pred_dist_relaxed, pred_var_relaxed, label_dist_relaxed)
            loss_cell = criterion_cell(pred_cell, label_cell)
            loss = loss_dist_displace + loss_dist_relaxed + loss_cell

            running_loss.update(loss.item()) 
            running_loss_dist_displace.update(loss_dist_displace.item(), label_dist_displace.size(0)) 
            running_loss_dist_relaxed.update(loss_dist_relaxed.item(), label_dist_relaxed.size(0)) 
            running_loss_cell.update(loss_cell.item(), label_cell.size(0)) 

            pred_dist_displace_list.append(pred_dist_displace.detach().cpu().numpy())
            label_dist_displace_list.append(label_dist_displace.detach().cpu().numpy())

            pred_dist_relaxed_list.append(pred_dist_relaxed.detach().cpu().numpy())
            label_dist_relaxed_list.append(label_dist_relaxed.detach().cpu().numpy())

            pred_cell_list.append(pred_cell.detach().cpu().numpy())
            label_cell_list.append(label_cell.detach().cpu().numpy())

    valid_loss = running_loss.get_average()
    valid_loss_dist_displace = running_loss_dist_displace.get_average()
    valid_loss_dist_relaxed = running_loss_dist_relaxed.get_average()
    valid_loss_cell = running_loss_cell.get_average()

    pred_dist_displace = np.concatenate(pred_dist_displace_list, axis=0)
    label_dist_displace = np.concatenate(label_dist_displace_list, axis=0)

    pred_dist_relaxed = np.concatenate(pred_dist_relaxed_list, axis=0)
    label_dist_relaxed = np.concatenate(label_dist_relaxed_list, axis=0)

    pred_cell = np.concatenate(pred_cell_list, axis=0).reshape(-1)
    label_cell = np.concatenate(label_cell_list, axis=0).reshape(-1)

    valid_mae_dist_displace = mean_absolute_error(pred_dist_displace, label_dist_displace)
    valid_mae_dist_relaxed = mean_absolute_error(pred_dist_relaxed, label_dist_relaxed)
    valid_mae_cell = mean_absolute_error(pred_cell, label_cell)

    model.train()

    return valid_loss, valid_loss_dist_displace, valid_loss_dist_relaxed, valid_loss_cell, valid_mae_dist_displace, valid_mae_dist_relaxed, valid_mae_cell

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_norm', type=int, default=150, help='max_norm for clip_grad_norm')
    parser.add_argument('--epochs', type=int, default=800, help='epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=800, help='steps_per_epoch')
    parser.add_argument('--early_stop_epoch', type=int, default=50, help='steps_per_epoch')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--transfer', type=bool, default=False)

    args = parser.parse_args()
    data_root = args.data_root
    num_workers = args.num_workers
    batch_size = args.batch_size
    max_norm = args.max_norm
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    early_stop_epoch = args.early_stop_epoch
    save_model = args.save_model
    transfer = args.transfer 

    train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train_DeepRelax')}, transform=reorder_transform)
    valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'val_DeepRelax')}, transform=reorder_transform)


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    dataset = data_root.split('/')[-1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f'DeepTransform_{dataset}_{timestamp}'
 

    device = torch.device('cpu')
    model = DeepTransform(hidden_channels=256, num_layers=8, num_rbf=128, cutoff=30.0, num_elements=118).to(device)
    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    
    if transfer == True:
        print("Loading pretrained model")

        model_path = './trained_model/model.pt'
        ema_helper.load_state_dict(torch.load(model_path))
        ema_helper.ema(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1.e-8)
    criterion_dist = DistanceL1Loss()
    criterion_cell = nn.L1Loss()

    running_loss = AverageMeter()
    running_loss_dist_displace = AverageMeter()
    running_loss_dist_relaxed = AverageMeter()
    running_loss_cell = AverageMeter()
    running_grad_norm = AverageMeter()
    running_best_mae = BestMeter("min")

    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    global_step = 0
    global_epoch = 0

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    break_flag = False
    model.train()

    for epoch in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1      

            data = data.to(device)
            pred_dist_displace, pred_var_displace, pred_dist_relaxed, pred_var_relaxed, pred_cell = model(data)
            label_dist_displace, label_dist_relaxed, label_cell \
                                            = get_edge_dist_displace(data), get_edge_dist_relaxed(data), data.cell_r
        
            loss_dist_displace = criterion_dist(pred_dist_displace, pred_var_displace, label_dist_displace)
            loss_dist_relaxed = criterion_dist(pred_dist_relaxed, pred_var_relaxed, label_dist_relaxed)
            loss_cell = criterion_cell(pred_cell, label_cell)
            loss = loss_dist_displace + loss_dist_relaxed + loss_cell
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
            )
            optimizer.step()

            ema_helper.update(model)

            running_loss.update(loss.item()) 
            running_loss_dist_displace.update(loss_dist_displace.item(), label_dist_displace.size(0)) 
            running_loss_dist_relaxed.update(loss_dist_relaxed.item(), label_dist_relaxed.size(0)) 
            running_loss_cell.update(loss_cell.item(), label_cell.size(0)) 
            running_grad_norm.update(grad_norm.item())

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                train_loss = running_loss.get_average()
                train_loss_dist_displace = running_loss_dist_displace.get_average()
                train_loss_dist_relaxed = running_loss_dist_relaxed.get_average()
                train_loss_cell = running_loss_cell.get_average()
                train_grad_norm = running_grad_norm.get_average()

                running_loss.reset()
                running_loss_dist_displace.reset()
                running_loss_dist_relaxed.reset()
                running_loss_cell.reset()
                running_grad_norm.reset()

                valid_loss, valid_loss_dist_displace, valid_loss_dist_relaxed, valid_loss_cell, \
                valid_mae_dist_displace, valid_mae_dist_relaxed, valid_mae_cell \
                                                = val(ema_helper.ema_copy(model), valid_loader, criterion_dist, criterion_cell, device)
                scheduler.step(valid_mae_dist_displace)

                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/epoch' : global_epoch,
                    'train/loss' : train_loss,
                    'train/loss_dist_displace' : train_loss_dist_displace,
                    'train/loss_dist_relaxed' : train_loss_dist_relaxed,
                    'train/loss_cell' : train_loss_cell,
                    'train/grad_norm' : train_grad_norm,
                    'train/lr' : current_lr,
                    'val/valid_loss' : valid_loss,
                    'val/valid_loss_dist_displace' : valid_loss_dist_displace,
                    'val/valid_loss_dist_relaxed' : valid_loss_dist_relaxed,
                    'val/valid_loss_cell' : valid_loss_cell,
                    'val/valid_mae_dist_displace' : valid_mae_dist_displace,
                    'val/valid_mae_dist_relaxed' : valid_mae_dist_relaxed,
                    'val/valid_mae_cell' : valid_mae_cell,
                }

                if valid_mae_dist_displace < running_best_mae.get_best():
                    running_best_mae.update(valid_mae_dist_displace)
                    if save_model:
                        torch.save(ema_helper.state_dict(), os.path.join("new_model", "model.pt"))
                else:
                    count = running_best_mae.counter()
                    if count > early_stop_epoch:
                        best_mae = running_best_mae.get_best()
                        print(f"early stop in epoch {global_epoch}")
                        print("best_mae: ", best_mae)
                        break_flag = True
                        break
                    
