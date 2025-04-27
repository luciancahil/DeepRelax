# %%
import math
import torch
from torch import nn
from torch_scatter import scatter
from graph_utils import ScaledSiLU, AtomEmbedding, RadialBasis
from graph_utils import cell_offsets_to_num, sinusoidal_positional_encoding, vector_norm
import torch.nn.functional as F

SOS_token = 1
class DeepRelax(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        num_layers=3,
        num_rbf=128,
        cutoff=6.,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        num_elements=83,
        d_model=128,
        max_atoms=0,
    ):
        super(DeepRelax, self).__init__()
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

        self.max_atoms = max_atoms

        self.decoder = Decoder(max_atoms)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data, target_tensor = None):
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

        # No. vec doesn't appear, X does.

        # Predict atom-wise displacements between the relaxed and unrelaxed structures 
        # The comment above is a lie.
        # actually predicts pair-wise atomic distances. For all atoms i, predict the distance between it and all atoms j.
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

        # Well, I guess that makes sense. It knows where the changes are because the edge_feat matrix is fed to it. I like this.
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



        # use [(n * (n-1)).item() for n in data.natoms] to figure out which pred_distance_displace goes where.
        # look at data .neighbors to see which edge, and by extension which pred_distance_relaxed, belongs where

        # pred_cell I know belongs with

        if(self.max_atoms == 0):
            return pred_distance_displace, pred_var_displace, pred_distance_relaxed, pred_var_relaxed, pred_cell
        else:
            return self.decoder(data, pred_distance_displace, pred_var_displace, pred_distance_relaxed, pred_var_relaxed, pred_cell, target_tensor = target_tensor)



class Decoder(nn.Module):
    def __init__(self, max_atoms):
        super().__init__()  # <-- this is missing!
        max_atoms = max(5, max_atoms)
        self.max_atoms = max_atoms
        self.attn_decoder = AttnDecoderRNN(max_atoms - 1, max_atoms-1)

        self.cell_projection_hidden = nn.Linear(9, max_atoms - 1)
        self.cell_projection_initial = nn.Linear(9, max_atoms - 1)

    def get_distance_as_tokens(self, data, num_samples, distance_tensor):
        if(distance_tensor == None):
            return None
        
        device = distance_tensor.device  # <- add this

        prev_distance_displace_index = 0
        distances_as_tokens = []
        for i in range(num_samples):
            cur_atoms = data.natoms[i]
            cur_pred_distance_index = prev_distance_displace_index + (cur_atoms * (cur_atoms - 1))
            
            cur_distances = distance_tensor[prev_distance_displace_index: cur_pred_distance_index]

            cur_tokens = torch.zeros(self.max_atoms, self.max_atoms - 1, device=device)

            cur_tokens[0: cur_atoms, 0:(cur_atoms - 1)] = torch.reshape(cur_distances, (cur_atoms, cur_atoms-1))

            distances_as_tokens.append(cur_tokens)

            prev_distance_displace_index = cur_pred_distance_index

        distances_as_tokens = torch.stack(distances_as_tokens, dim=0)

        return distances_as_tokens

    def forward(self, data, pred_distance_displace, pred_var_displace, pred_distance_relaxed, pred_var_relaxed, pred_cell, target_tensor=None):
        num_samples = len(data.natoms)

        # reshape the pred_distance_displace into max_atom * max_atom - 1
        
        cell_hidden = self.cell_projection_hidden(torch.reshape(pred_cell, (-1, 9)))

        cell_initial = self.cell_projection_initial(torch.reshape(pred_cell, (-1, 9)))

        distances_as_tokens = self.get_distance_as_tokens(data, num_samples, pred_distance_displace)

        target_tensor_tokens = self.get_distance_as_tokens(data, num_samples, target_tensor)
        
        
        decoded_distance_displace = self.attn_decoder(distances_as_tokens, cell_hidden, cell_initial, target_tensor = target_tensor_tokens)[0]

        prev_distance_displace_index = 0

        new_pred_distances = torch.zeros_like(pred_distance_displace, device=decoded_distance_displace.device)
        for i in range(num_samples):
            cur_atoms = data.natoms[i]
            
            cur_pred_distance_index = prev_distance_displace_index + (cur_atoms * (cur_atoms - 1))

            cur_distances = decoded_distance_displace[i][0:cur_atoms,0:(cur_atoms-1)].reshape(-1)
            
            new_pred_distances[prev_distance_displace_index:cur_pred_distance_index] = cur_distances
            prev_distance_displace_index = cur_pred_distance_index  


        return new_pred_distances, pred_var_displace, pred_distance_relaxed, pred_var_relaxed, pred_cell

        # I'm really nervous how long this could take.

        # I could try to have each vector in pred_distance represent all interatomic distances for a single atom.

        # 20 tokens of size 19 ain't that bad.

        # what do I do with pred_distance_relaxed?

        # there is SO MUCH variance in the number of neighbours.

        # Maybe I'll just rest up a bit?


        # hold on. I really just need to figure out the max for both natoms and neighbours.

        # Not only that, go through edge_index for each, and see wich number recieves the most edges.

        # So in short, go through each graph.

        # for each, find the number of atoms in the graph.

        # find the atom with the most edges going towards it. Record the count, and subtract (natoms - 1).

        # for both of the above, keep track of the max

        # from that, figure it out.

        # but for now, I am just so dead.

        # also, check. The first natoms * (natoms -1 ) for both in the ground trugh should be the same? Check that


        # another idea, just let the network peak at the cell lattice, potentially with teacher forcing.
        pass


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length = 20):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.max_length = max_length

    # Seems like all I really have to do is come up with a way to come up with encode_hidden. Maybe an output of the
    def forward(self, encoder_outputs, encoder_hidden, initial, is_training=False, target_tensor=None):
        decoder_input = initial.unsqueeze(dim = 1)
        decoder_hidden = encoder_hidden.unsqueeze(dim = 0)
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):

            decoder_hidden = decoder_hidden.contiguous()
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, isTraining = is_training
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # remove this top i stuff
                # Without teacher forcing: use its own predictions as the next input
                decoder_input = decoder_output

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs, isTraining = False):
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((input, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    

class Variator(nn.Module):
    def __init__(self, input_size, latent_size = 100, output_size = None):
        super(Variator, self).__init__()
        if(output_size == None):
            output_size = input_size
        self.encode_mu = nn.Linear(input_size, latent_size)
        self.encode_log_var = nn.Linear(input_size, latent_size)
        self.decode_layer = nn.Linear(latent_size, output_size)    

    def encode(self, x):
        log_var = self.encode_log_var(x)
        x = self.encode_mu(x)
        return x, log_var

    def decode(self, x):
        x = self.decode_layer(x)

        return x

    def reparameterize(self, x, log_var):
        # Get standard deviation
        std = torch.exp(log_var)
        # Returns random numbers from a normal distribution
        eps = torch.randn_like(std)
        # Return sampled values
        return eps.mul(std).add_(x)

    def forward(self, x, isTraining = False):
        mu, log_var = self.encode(x)

        if(isTraining):
            x = self.reparameterize(mu, log_var)
            x = self.decode(x)
        else:
            x = self.decode(mu)
        
        return x, mu, log_var

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
- Combine Crystal Train and train.py
- Turn each node's distances into a sequence. So we have 32 x 20 sequences. Lovely!
"""