"""
python train_process.py --data_root ./datasets/Data_MP_XMnO/cifs_xmno --num_workers 4 --batch_size 32 --steps_per_epoch 800

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from DeepRelax import DeepRelax
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

def get_target_node_only(graph, j, edge_data):

    target_list = [graph.x[j].item() / 118, -graph.x[j].item() / 118, graph.x[j].item() / 118]

    return str(target_list).replace(" ", "")[1:-1]


def get_target_gravity(graph, j, edge_data):
    shift = torch.zeros(3)

    for edge in edge_data:
        mass = graph.x[edge[0]] / 118
        dist = edge[1]

        reciprocal = 1 / (dist ** 3)

        displacement = torch.tensor(edge[-3:])

        shift += displacement * reciprocal * mass


    target_list = graph.pos_u[j] + shift

    return str(target_list.tolist()).replace(" ", "")[1:-1]

def get_actual_target(graph, j, edge_data):
    return str(graph.pos_r[j].tolist()).replace(" ","")[1:-1]

def get_shifted_by_node(graph, j, edge_data):

    target_list = torch.tensor([graph.x[j].item() / 118, -graph.x[j].item() / 118, graph.x[j].item() / 118]) + graph.pos_u[j]

    return str(target_list.tolist()).replace(" ", "")[1:-1]


def get_target(graph, j, edge_data):
    return get_target_gravity(graph, j, edge_data)

def process(loader,file):
    idx = -1
    for batch in loader:
        pos = batch.pos_u
        cell = batch.cell_u
        neighbors = batch.neighbors

        cell_offsets = batch.cell_offsets
        cell_offsets_unsqueeze = cell_offsets.unsqueeze(1).float()
        abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)    
        edge_index = batch.edge_index
        j, i = edge_index

        vecs = (pos[j] + (cell_offsets_unsqueeze @ abc_unsqueeze).squeeze(1)) - pos[i]
        prev_neighbor = 0

        for i in range(len(batch)):
            edge_dist = vecs[prev_neighbor:prev_neighbor + neighbors[i]]


            graph = batch[i]
            idx += 1

            if(idx % 1000 == 0):
                print(idx)


            file.write("{}--{}--{}--{}\n".format(idx,len(graph.x),str(graph.cell_u.tolist()[0])[1:-1],str(graph.cell_r.tolist()[0])[1:-1]))

            # make an array.
            edge_data = [None] * len(graph.x)
            # the key of the array is the source node.
            # the object is another list.

            # In that list, is a tensor, containing all info I need for the graph.

            # just replace all the prints with file.writes, and I'm done!
            for j in range(graph.edge_index.shape[1]):
                source = graph.edge_index[0][j]
                dest = graph.edge_index[1][j]
                dist_arr = edge_dist[j]
                dist = torch.sqrt(sum([n*n for n in edge_dist[j]]))
                cell_o = graph.cell_offsets[j]

                if(edge_data[source] == None):
                    edge_data[source] = []
                
                edge_data[source].append((dest.item(), dist.item(), cell_o[0].item(), cell_o[1].item(), cell_o[2].item(), dist_arr[0].item(), dist_arr[1].item(), dist_arr[2].item()))

            for j in range(len(edge_data)):
                edge_data[j] = sorted(edge_data[j], key=lambda a:a[-1])
                edge_data[j] = sorted(edge_data[j], key=lambda a:a[-2])
                edge_data[j] = sorted(edge_data[j], key=lambda a:a[-3])
                edge_data[j] =  sorted(edge_data[j], key=lambda a:a[1])
                edge_data[j] =  sorted(edge_data[j], key=lambda a:a[0])
                

            for j in range(len(graph.x)):
                x = graph.x[j]
                pos_u = str(graph.pos_u[j].tolist()).replace(" ","")[1:-1]
                pos_r = get_target(graph, j, edge_data[j])

                edge_string = str(edge_data[j][0]).replace(" ", "").replace("(","").replace(")", "")

                for edge in edge_data[j][1:]:
                    edge_string +=";" + str(edge).replace(" ", "").replace("(","").replace(")", "")

                file.write("{}:0,0,0:{}:{}:{}\n".format(x, pos_u, pos_r, edge_string))

            prev_neighbor += neighbors[i]



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

    train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train_DeepRelax')})
    valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'val_DeepRelax')})

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    file = open("raw/Bulk_node_gravity.txt", mode='w')

    process(train_loader, file)
    process(valid_loader, file)
"""
TODO:
The start should be the atomic number, NOT the atomic symbol.


"""