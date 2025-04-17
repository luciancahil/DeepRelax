# getitem must return: None, input_graph, target_graph, null, null.
import os
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
import numpy as np
import re
import pickle
import lmdb


class CrystalGraphDataset(Dataset):
    def __init__(self, datasetName, embedding = False):
        super().__init__()
        self.dataset_path = "dataset"
        self.processed_path = os.path.join(self.dataset_path , "processed", datasetName)
        self.raw_path = os.path.join(self.dataset_path, "raw")

        self.lmdb_path = os.path.join(self.processed_path, datasetName + ".lmdb")
        self.embedding = embedding
        

        if(os.path.exists(self.processed_path)):
            print("Processings skipped because diectory dataset/processed/{} exists. If you wish to process raw data, please delete that folder.".format(datasetName))
        else:
            print("Begin Processing")
            self.process(datasetName)
        
        self._load_lmdb()


    def _load_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            print("hello!")
            self.len = pickle.loads(txn.get(b'len'))  # Load total dataset length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            item_bytes = txn.get(str(idx).encode())
            if item_bytes is None:
                return None, None, None, None, None
            input_graph = pickle.loads(item_bytes)
        return input_graph

    def __len__(self):
        return self.len
            
           
    def process(self,datasetName, embedding = False):
        self.process_features(datasetName)
    
    def process_features(self, datasetName):
        os.mkdir(self.processed_path)

        raw_file = open(os.path.join(self.raw_path, "{}.txt".format(datasetName)), mode='r')
        defining = raw_file.readline()
        idx = 0
        all_starting = torch.empty(0, 3)
        all_ending = torch.empty(0, 3)
        element_set = set()
        # the number of samples we reject
        rejected = 0

        os.makedirs(os.path.dirname(self.lmdb_path), exist_ok=True)
        env = lmdb.open(self.lmdb_path, map_size=3*10**9)  # Adjust map_size as needed
        element_to_number = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
            "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
            "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
            "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
            "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
            "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
            "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
            "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
            "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
            "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
            "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
        }


        print("Helre!")
        with env.begin(write=True) as txn:
            idx = 0
            rejected = 0

            while(not(defining == "" or defining == "\n")):

                if(idx % 100 == 0):
                    print(idx)    
                
                parts = defining.strip().split("--")
                num_nodes = int(parts[1])
                cid = parts[0]
                
                # this is what I'm dealing with:
                # '[5.93378261 0.         0.        ],[2.96689131 5.13880648 0.        ],[ 0.          0.         99.06947929]'
                lattice = torch.tensor([[float(num) for num in re.split(r' +', l)[1:-1]] for l in parts[2].replace("["," ").replace("]"," ").split(",")])
                start_pos_list = []
                end_pos_list = []

                features_list = []
                elements = []
                edges = []
                edge_attr = torch.empty(0, 3)
                cell_offsets = [[0,0,0]] * (num_nodes * (num_nodes - 1))
                
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        
                        if(i == j):
                            continue
                        
                        edges.append([i, j])
                

                # read data
                for i in range(num_nodes):
                    node_line = raw_file.readline()
                    parts = node_line.split(":")
                    elements.append(parts[0])
                    start_pos_list.append([float(feature) for feature in parts[2].split(",")])
                    end_pos_list.append([float(feature) for feature in parts[3].split(",")])

                    features_list.append([float(p) for p in parts[1].split(",")])

                    cur_edges = parts[4].strip().split(";")

                    for edge in cur_edges:

                        edge_parts = edge.split(",")
                        if(edge_parts[0] == ""):
                            continue
                            
                        cell_offset = [float(edge_parts[2]), float(edge_parts[3]), float(edge_parts[4])]
                        cell_offsets.append(cell_offset)
                        edge_target = int(edge_parts[0])
                        edge_data = torch.tensor([float(num) for num in edge_parts[1:]][4:])

                        edges.append([i, edge_target])
                        edge_attr = torch.concat((edge_attr, edge_data.unsqueeze(0)), dim = 0)
                

                start_pos_list = torch.tensor(start_pos_list, dtype=torch.float)

                edges = torch.tensor(edges)
                edges = torch.transpose(edges, 0, -1)
                edges = edges.long()
                features_list = torch.tensor(features_list, dtype=torch.float)
                end_pos_list = torch.tensor(end_pos_list, dtype=torch.float)
                elements = torch.tensor([element_to_number[atom] for atom in elements])

                # move everything down so that the top of the surface has a height of 0.
                # 3rd because first 2 is either CO or H. So 2nd atom for H, but usually same level
                scale = start_pos_list[:,2].sort().values[-3]

                start_pos_list[:,2] -= scale
                end_pos_list[:, 2] -= scale

                dist_pos = (end_pos_list-start_pos_list)[:,0:2]
                basis_vector =  np.linalg.inv(torch.transpose(lattice[0:2,0:2], 0, 1))


                normed_basis = torch.transpose((torch.tensor(basis_vector, dtype=torch.float) @ torch.transpose(dist_pos, 0, 1)*100), 0, 1)  # Transform the vector
                
                if (max(normed_basis.norm(dim=1).tolist()) > 0.7):
                    idx += 1
                    defining = raw_file.readline()
                    rejected += 1
                    continue
                
                all_starting = torch.concat((all_starting, start_pos_list), dim=0)
                all_ending = torch.concat((all_ending, end_pos_list), dim=0)
                natoms = torch.tensor(len(elements))
                pbc = torch.tensor([True, True, True])
                cell_offsets = torch.tensor(cell_offsets)

                mask = [False] * edges.shape[1]

                mask[0:(len(elements) * (len(elements) - 1))] = [True]*(len(elements) * (len(elements) - 1))
                mask = torch.tensor(mask)

                lattice = lattice.unsqueeze(dim=0)
                input_graph = Data(x=elements, edge_index=edges, cell_u=lattice, pos_u=start_pos_list, cell_r=lattice, pos_r=end_pos_list,\
                                    edge_attr=edge_attr, cell_offsets=cell_offsets, natoms=natoms, pbc=pbc, mask=mask, cif_id=cid)

                txn.put(str(idx - rejected).encode(), pickle.dumps((input_graph)))


            
                defining = raw_file.readline()
                idx += 1
            

            txn.put(b'len', pickle.dumps(idx - rejected))  # Save dataset length

            env.sync()
            print(f"Rejected {rejected} samples. Dataset saved to LMDB.")