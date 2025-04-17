# %%
import math
import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch_scatter import scatter
from graph_utils import ScaledSiLU, AtomEmbedding, RadialBasis
from graph_utils import cell_offsets_to_num, sinusoidal_positional_encoding, vector_norm
from torch.nn.utils.rnn import pad_sequence

class DeepTransform(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        transformer_dim = 256,
        num_layers=3,
        transformer_layer = 7,
        num_rbf=128,
        cutoff=6.,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        num_elements=83,
        d_model=128,
        num_attn_heads = 8,
    ):
        super(DeepTransform, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.d_model = d_model

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_layers.append(
                MessagePassing(hidden_channels, num_rbf + d_model)
            )
            self.update_layers.append(MessageUpdating(hidden_channels))


        self.graph_transformer_encoder = CrystalTransform(transformer_dim, num_attn_heads)

        self.lattice_ecoder = CrystalTransform(transformer_dim, num_attn_heads)
        
        self.distance_encoder = CrystalTransform(transformer_dim, num_attn_heads)

        self.lattice_decoder = CrystalTransform(transformer_dim, num_attn_heads)

        self.distance_decoder = CrystalTransform(transformer_dim, num_attn_heads)

        self.main_attention_layers = nn.ModuleList()

        # number of heads and main layers are equal
        for i in range(num_attn_heads):
            self.main_attention_layers.append(CrystalTransform(transformer_dim, num_attn_heads))
        

        # In keeping with GPT tradition, the number of attention layers will equal # the number of layers

        # In the main skeleton

        # I need 7 anyways: E/D for the crystal, pair, and edge, and the encoding stuff that comes out of the graphs.

        # So for now, Encode graphs. Encode code graphs, encode lattice, decode lattice, encode distance, decode distance, decode displacement.


        self.dist_displace_branch = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU()
        )
        self.dist_relaxed_branch = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU()
        )
        self.cell_branch = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU()
        )

        self.lin_edge_displace = nn.Sequential(
            nn.Linear(num_rbf + d_model, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU()
        )
        self.out_distance_displace = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, 2),
        )

        self.lin_edge_relaxed = nn.Sequential(
            nn.Linear(num_rbf + d_model, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU()
        )
        self.out_distance_relaxed = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, 2),
        )

        self.lin_cell = nn.Sequential(
            nn.Linear(9, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU()
        )
        self.out_cell = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, 9),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data):
        # position of each atom unrelaxed
        pos = data.pos_u

        # lattice vectors
        cell = data.cell_u
        
        # cell_offsets = data.cell_offsets
        cell_offsets = data.cell_offsets
        edge_index = data.edge_index

        neighbors = data.neighbors

        # which atom belongs to which batch. batch[i] = j means atom i is part of batch j.
        batch = data.batch
        z = data.x.long()
        assert z.dim() == 1 and z.dtype == torch.long

        j, i = edge_index
        cell_offsets_unsqueeze = cell_offsets.unsqueeze(1).float()

        # okay, the first neighbours[0] will be equal to cell[0], and repeat for every element in both cell and neighbours
        abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)

        # this is why nobody likes clever programmers.
        # pos[i] creates a matrix with as many rows as i, where pos[i][N] = pos[i[N]].
        # this tries to do some sort of modifiication.
        # why are parenthates there? Addition is associative, and even otherwise, you'd do it like that anyways.
        # Okay, so the way this works is by finding the vector from atom 1 to atom 2. 
        # oh, my god. This is literally just the vector between two atoms in an edge!
        # that multiplication stuff is just meant to account for cell differences!

        # why would you make that so complicated?
        vecs = (pos[j] + (cell_offsets_unsqueeze @ abc_unsqueeze).squeeze(1)) - pos[i]
        
        # calculates the length of the vector, and divide, so that all edge_vectors are normalized (magnitude 1)
        edge_dist = vector_norm(vecs, dim=-1)
        edge_vector = -vecs/edge_dist.unsqueeze(-1)

        edge_rbf = self.radial_basis(edge_dist)  # rbf * evelope
        cell_offsets_int = cell_offsets_to_num(cell_offsets)
        cof_emb = sinusoidal_positional_encoding(cell_offsets_int, d_model=self.d_model)
        edge_feat = torch.cat([edge_rbf, cof_emb], dim=-1)

        x = self.atom_emb(z)

        # I don't quite know what this means? 
        # It's a 3D tensor, With dimesions as x but with a 3 squeezed in there for some reason.
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        
        #### Interaction blocks ###############################################
        breakpoint()
        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](
                x, vec, edge_index, edge_feat, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)
            x = x + dx
            vec = vec + dvec

        # Okay, If I want to figure out why and how this works, I don't know. 

        # I really need to figure out information that gets spat out.

        # Maybe it's just dvec and dx?

        # Predict atom-wise displacements between the relaxed and unrelaxed structures 

        # I got it now. I need to predict both mean and variance for boht inter and intra cellurlar distance.

        # So first, I gotta group up everything into a 3D tensor, with batch x atoms x dim.

        # I know how many tokens I need in every case, so shouldn't be a problem there.

        # Then run, until dropping out as need be.

        # Once that's done, we can move onto the next step.

        # The first time, it isn't gonna be so hard, since we need 3x3 each time.

        # The next, gonna be a little tricky.

        # use the dropout thing.

        # Then, once I'm done, I've gotta do some nasty splitting to make sure everything is where I want it to be.

        # First, group everything into a 3D tensor, and add masks to the padding
        grouped_X = [x[data.batch == i] for i in range(32)]
        grouped_pos = [pos[data.batch == i] for i in range(32)]

        # Pad sequences to max length
        padded_X = torch.zeros(len(grouped_X), 20, grouped_X[0].shape[1])
        padded_pos = torch.zeros(len(grouped_X), 20, grouped_pos[0].shape[1])

        for i in range(len(grouped_X)):
            padded_X[i][0:grouped_X[i].shape[0]] = torch.add(padded_X[i][0:grouped_X[i].shape[0]] , grouped_X[i])
            padded_pos[i][0:grouped_pos[i].shape[0]] = torch.add(grouped_pos[i], padded_pos[i][0:grouped_pos[i].shape[0]] )


        # Create an attention mask: 1 for real data, 0 for padding
        mask = torch.ones(padded_X.shape[:2])  # Shape: (32, max_N)
        for i, group in enumerate(grouped_X):
            mask[i, :group.shape[0]] = 0  # Set padding positions to 0

        padded_X = padded_X + encode_position(padded_X, padded_pos)

        #TODO: why do I have so many dimensions? Fix that

        # Then, just go through it all.

        breakpoint()
        self.graph_transformer_encoder(padded_X, mask)
        # next step is to use mha, combined with up project and down project. Gotta make that first.




        # Don't use MHA to calculate the edge attrs. Just use the outputs of pair-wise distance and cell_lattice, combined with cell_offsets_unsqueeze

        # specifically, get all of them into a matrix, and then just go.

        # cell lattice needs to be in a n_total_edges x 3 x 3.
        #         vecs = (pos[j] + (cell_offsets_unsqueeze @ abc_unsqueeze).squeeze(1)) - pos[i]

        # new positions needs to be in a n_total_
        mask_dispalce = data.mask
        edge_index_displace = edge_index[:, mask_dispalce]
        edge_feat_displace = edge_feat[mask_dispalce]
        edge_feat_displace = self.lin_edge_displace(edge_feat_displace)
        j, i = edge_index_displace
        x_dist_displace = self.dist_displace_branch(x)
        dist_feat_displace = torch.cat([x_dist_displace[i], x_dist_displace[j], edge_feat_displace], dim=-1)
        pred_distance_var_displace = self.out_distance_displace(dist_feat_displace)
        pred_distance_displace, pred_var_displace = torch.split(pred_distance_var_displace, 1, -1)
        pred_distance_displace, pred_var_displace = torch.relu(pred_distance_displace).squeeze(-1), pred_var_displace.squeeze()

        # Predict pair-wise distances within the relaxed structure
        edge_feat_relaxed = self.lin_edge_relaxed(edge_feat)
        j, i = edge_index
        x_dist_relaxed = self.dist_relaxed_branch(x)
        dist_feat_relaxed = torch.cat([x_dist_relaxed[i], x_dist_relaxed[j], edge_feat_relaxed], dim=-1)
        pred_distance_var_relaxed = self.out_distance_relaxed(dist_feat_relaxed)
        pred_distance_relaxed, pred_var_relaxed = torch.split(pred_distance_var_relaxed, 1, -1)
        pred_distance_relaxed, pred_var_relaxed = torch.relu(pred_distance_relaxed).squeeze(-1), pred_var_relaxed.squeeze(-1)
  
        # Predict the cell of the relaxed structure
        x_cell = self.cell_branch(x)
        g_feat_cell = scatter(x_cell, batch, dim=0) 
        cell_feat = self.lin_cell(cell.view(-1, 9))
        pred_cell = self.out_cell(torch.cat([g_feat_cell, cell_feat], dim=-1)).view(-1, 3, 3) + cell

        return pred_distance_displace, pred_var_displace, pred_distance_relaxed, pred_var_relaxed, pred_cell

def encode_position(x, pos):
    # sin(x/4000^(i/dim)) for first 
    # sin(y/4000^(i/dim) + 2pi/3) for middle 3rd
    # sin(z/4000^(i/dim) + 4pi/3) for last 3rd
    
    #TODO: THIS
    dim = x.shape[-1]
    third = int(dim/3)
    positional_matrix = torch.zeros([3, dim])

    # addd 1's
    positional_matrix[0][0:third] = 1
    positional_matrix[1][third: 2*third] = 1
    positional_matrix[2][2*third:] = 1

    positional_matrix = pos @ positional_matrix

    # we need 1/ [4000 ^ (3i/dim)] = 1/[e^(ln(4000))]^(3i/dim) = e^[-3i*ln(4000)/dim]
    constant = -1* math.log(4000) / dim

    scale = torch.tensor([[[i * constant for i in range(positional_matrix.shape[2])] for j in range(positional_matrix.shape[1])] for k in range(positional_matrix.shape[0])])

    scale = torch.exp(scale)

    positional_matrix = torch.div(positional_matrix, scale)

    positional_matrix[:,:,third:2*third] += 2*math.pi/3

    positional_matrix[:,:,2*third:] += 4*math.pi/3    


    return positional_matrix




class CrystalTransform(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(CrystalTransform, self).__init__()

        self.attention = MultiheadAttention(embedding_size, num_heads,batch_first=True)
        self.up_project = nn.Linear(embedding_size, 4 * embedding_size)
        self.down_project = nn.Linear(4*embedding_size, embedding_size)
        self.activation = nn.LeakyReLU()
        self.embedding_zie = embedding_size
    



    def forward(self, x, mask):
        breakpoint()
        attn = self.attention(x, x, x, key_padding_mask=mask)
        x = x + attn
        x = self.up_project(x)
        x = self.activation(x)
        x = self.down_project(x)
        x = self.activation
        
        return x

class MessagePassing(nn.Module):
    def __init__(
        self,
        hidden_channels,
        edge_feat_channels,
    ):
        super(MessagePassing, self).__init__()

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels*3),
        )
        self.edge_proj = nn.Linear(edge_feat_channels, hidden_channels*3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        j, i = edge_index

        rbf_h = self.edge_proj(edge_rbf)

        x_h = self.x_proj(x)
        x_ji1, x_ji2, x_ji3 = torch.split(x_h[j] * rbf_h * self.inv_sqrt_3, self.hidden_channels, dim=-1)

        vec_ji = x_ji1.unsqueeze(1) * vec[j] + x_ji2.unsqueeze(1) * edge_vector.unsqueeze(2)
        vec_ji = vec_ji * self.inv_sqrt_h

        d_vec = scatter(vec_ji, index=i, dim=0, dim_size=x.size(0)) 
        d_x = scatter(x_ji3, index=i, dim=0, dim_size=x.size(0)) 
        
        return d_x, d_vec
    
class MessageUpdating(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.
        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1

        return dx, dvec
# %%


"""
TODO: 

Figure out how all the stuff works for prediction outputs.

Standardize everything.

Then have a go at it.

How can I standardize anything? I'm so confused.

Tell me about data! Why does he wear the mask.

Welp. It's there somehow in the data. As soson as it's unpickled, it's there
"""