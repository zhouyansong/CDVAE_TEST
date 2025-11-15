from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits
from torch_scatter import scatter, scatter_min
from lrcdvae.pl_modules.comenet.features import angle_emb, torsion_emb
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Embedding
import math
from math import sqrt
try:
    import sympy as sym
except ImportError:
    sym = None
from lrcdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from lrcdvae.common.data_utils import (
    get_pbc_distances,
    frac_to_cart_coords,
    radius_graph_pbc_wrapper,
    radius_graph_pbc,
    conditional_grad,
    get_k_index_product_set,
    get_k_voxel_grid,
    lattice_params_to_matrix_torch,
    x_to_k_cell,
)
from lrcdvae.pl_modules.ewald_block import EwaldBlock
from lrcdvae.pl_modules.gemnet.layers.base_layers import Dense, ResidualLayer
from lrcdvae.pl_modules.gemnet.layers.embedding_block import AtomEmbedding, EdgeEmbedding
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock

def swish(x):
    return x * torch.sigmoid(x)

class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


# class HeteroLinear(torch.nn.Module):

#     def __init__(self, in_channels: int, out_channels: int,
#                  num_tags: int, **kwargs):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.lins = torch.nn.ModuleList([
#             Linear(in_channels, out_channels, **kwargs)
#             for _ in range(num_tags)
#         ])

#         self.reset_parameters()

#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()

#     def forward(self, x: Tensor, node_tag: Tensor) -> Tensor:
#         """"""
#         out = x.new_empty(x.size(0), self.out_channels)
#         for i, lin in enumerate(self.lins):
#             mask = node_tag == i
#             out[mask] = lin(x[mask], None)
#         return out


class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
    ):
        super(TwoLayerLinear, self).__init__()
        # if hetero:
        #     self.lin1 = HeteroLinear(in_channels, middle_channels, num_tags=3, bias=bias)
        #     self.lin2 = HeteroLinear(middle_channels, out_channels, num_tags=3, bias=bias)
        # else:
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)

        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish, num_elements=100):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        # Support up to 95 elements (extended periodic table)
        self.emb = Embedding(num_elements, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            middle_channels,
            num_radial,
            num_spherical,
            num_layers,
            output_channels,
            act=swish,
            inits='glorot',
    ):
        super(SimpleInteractionBlock, self).__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)
        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, middle_channels, hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, middle_channels, hidden_channels)

        # Dense transformations of input messages.
        # if hetero:
        #     self.lin = HeteroLinear(hidden_channels, hidden_channels, num_tags=3)
        #     self.lins = torch.nn.ModuleList()
        #     for _ in range(num_layers):
        #         self.lins.append(HeteroLinear(hidden_channels, hidden_channels, num_tags=3))
        #     self.final = HeteroLinear(hidden_channels, output_channels, num_tags=3, weight_initializer=inits)
        # else:
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.lin(x))

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))

        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        return h


class ComENet(nn.Module):
    def __init__(
            self,
            num_targets,
            otf_graph=False,
            use_pbc=True,
            hidden_channels=128,
            middle_channels=64,
            out_channels=1,
            num_blocks=4,
            num_radial=3,
            num_spherical=7,
            num_layers=4,
            max_num_neighbors=50,
            cutoff=8.0,
            num_output_layers=3,
            hetero=False,
            use_ewald=True,
            ewald_hyperparams=None,
            atom_to_atom_cutoff=None,
            readout='mean',  # 新增参数
            regress_forces=False,  # 添加这一行
            
    ):
        super(ComENet, self).__init__()
        self.num_targets = num_targets
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.num_blocks = num_blocks
        self.atom_to_atom_cutoff = atom_to_atom_cutoff
        self.use_ewald = use_ewald and ewald_hyperparams is not None
        self.use_atom_to_atom_mp = atom_to_atom_cutoff is not None
        self.readout = readout  # 存储readout参数
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.regress_forces = regress_forces  # 添加这一行
        self.max_num_neighbors = max_num_neighbors
        
        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act
        # ========== 核心特征提取 ==========
        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        # ========== 原子嵌入(用于ComENet短程交互) ==========
        self.emb = EmbeddingBlock(hidden_channels, act)

        # ========== Ewald 长程消息传递设置 ==========
        if self.use_ewald:
            if self.use_pbc:
                # PBC情况: 使用离散k-空间索引
                self.num_k_x = ewald_hyperparams["num_k_x"]
                self.num_k_y = ewald_hyperparams["num_k_y"]
                self.num_k_z = ewald_hyperparams["num_k_z"]
                self.delta_k = None
            else:
                # 非PBC: 使用连续k-空间体素网格
                self.k_cutoff = ewald_hyperparams["k_cutoff"]
                self.delta_k = ewald_hyperparams["delta_k"] # 体素分辨率
                self.num_k_rbf = ewald_hyperparams["num_k_rbf"] # 径向基函数数量
            
            self.downprojection_size = ewald_hyperparams["downprojection_size"]
            # Number of residuals in update function
            self.num_hidden = ewald_hyperparams["num_hidden"] # Ewald更新函数中的残差块数
            # 是否detach Ewald梯度(对于encoder建议False,因为需要端到端训练)
            self.detach_ewald = ewald_hyperparams.get("detach_ewald", False)
        
        # ========== k-空间结构初始化 ==========
        if self.use_ewald:
            if self.use_pbc:
                # Get the reciprocal lattice indices of included k-vectors
                (
                    self.k_index_product_set,
                    self.num_k_degrees_of_freedom,
                ) = get_k_index_product_set(
                    self.num_k_x,
                    self.num_k_y,
                    self.num_k_z,
                )
                self.k_rbf_values = None
                self.delta_k = None

            else:
                # Get the k-space voxel and evaluate Gaussian RBF (can be done at
                # initialization time as voxel grid stays fixed for all structures)
                (
                    self.k_grid,
                    self.k_rbf_values,
                    self.num_k_degrees_of_freedom,
                ) = get_k_voxel_grid(
                    self.k_cutoff,
                    self.delta_k,
                    self.num_k_rbf,
                )
            # Ewald专用原子嵌入(借用GemNet架构,与ComENet的emb独立)
            # 原因: Ewald需要不同的特征空间来处理长程相互作用
            self.atom_emb = AtomEmbedding(hidden_channels)
            # Downprojection layer, weights are shared among all interaction blocks
            self.down = Dense(
                self.num_k_degrees_of_freedom,
                self.downprojection_size,
                activation=None,
                bias=False,
            )
            self.ewald_blocks = torch.nn.ModuleList(
                [
                    EwaldBlock(
                        self.down,
                        hidden_channels,  # Embedding size of short-range GNN
                        self.downprojection_size,
                        self.num_hidden,  # Number of residuals in update function
                        activation="silu",
                        use_pbc=self.use_pbc,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(self.num_blocks)
                ]
            )
        # ========== atom-to-atom 长程消息传递(可选) ==========
        if self.use_atom_to_atom_mp:
            self.atom_emb = AtomEmbedding(hidden_channels)
            if self.use_pbc:
                self.max_neighbors_at = int(
                    (self.atom_to_atom_cutoff / 6.0) ** 3 * 50
                )
            else:
                self.max_neighbors_at = 100
            self.interactions_at = torch.nn.ModuleList(
                [
                    InteractionBlock(
                        hidden_channels,
                        200,  # num Gaussians
                        256,  # num filters
                        self.atom_to_atom_cutoff,
                    )
                    for i in range(self.num_blocks)
                ]
            )
            self.distance_expansion_at = GaussianSmearing(
                0.0, self.atom_to_atom_cutoff, 200
            )
        else:
            self.max_neighbors_at = None
            self.distance_expansion_at = None
        # 跳跃连接归一化因子
        # 作用: 防止多路信息叠加导致梯度爆炸
        # 公式: factor = 1 / sqrt(1 + n_long_range_paths)
        self.skip_connection_factor = (
            1.0 + float(self.use_ewald) + float(self.use_atom_to_atom_mp)
        ) ** (-0.5)

        # ========== ComENet Interaction Blocks (Core Architecture) ==========
        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )
        self.lins = torch.nn.ModuleList()
        # if hetero:
        #     for _ in range(num_output_layers):
        #         self.lins.append(HeteroLinear(hidden_channels, hidden_channels, num_tags=3))
        #     self.lin_out = HeteroLinear(hidden_channels, num_targets, num_tags=3, weight_initializer='zeros')
        # else:
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, num_targets)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()
        
    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        batch = data.batch
        z = data.atom_types.long()
        num_nodes = data.atom_types.size(0)
        batch_size = int(batch.max()) + 1

        # ========== Graph Construction (OTF or pre-computed) ==========
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc_wrapper(
                data, self.cutoff, self.max_num_neighbors, data.num_atoms.device
            )
            data.edge_index = edge_index
            data.to_jimages = cell_offsets
            data.num_bonds = neighbors

        pos = frac_to_cart_coords(
            data.frac_coords,
            data.lengths,
            data.angles,
            data.num_atoms)

        out = get_pbc_distances(
            pos,
            data.edge_index,
            data.lengths,
            data.angles,
            data.to_jimages,
            data.num_atoms,
            data.num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True
        )
        
        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]
        j, i = edge_index
        vecs = out["distance_vec"]


        # ========== 嵌入初始化 ==========
        # ComENet的短程嵌入
        x = self.emb(z)

        # Ewald长程嵌入(独立的特征空间)
        if self.use_ewald:
            # If Ewald MP is used, we have to create atom embeddings borrowing
            # the atomic embedding block from the GemNet architecture
            h_ewald = self.atom_emb(data.atom_types.long())
            dot = None  # These will be computed in first Ewald block and then passed
            sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
            pos_detach = pos.detach() if self.detach_ewald else pos
        edge_index_at, cell_offsets_at, neighbors_at = radius_graph_pbc(
                pos, data.lengths, data.angles, data.num_atoms, 
                self.atom_to_atom_cutoff, self.max_neighbors_at,
                device=data.num_atoms.device
            )
        out_at = get_pbc_distances(
                pos, edge_index_at, data.lengths, data.angles,
                cell_offsets_at, data.num_atoms, neighbors_at,
                coord_is_cart=True,
                return_offsets=True,
                return_distance_vec=True
            )
        edge_index_at = out_at["edge_index"]
        edge_weight_at = out_at["distances"]
        edge_attr_at = self.distance_expansion_at(edge_weight_at)
        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.linalg.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.linalg.cross(-pos_ji, pos_in0)
        plane2 = torch.linalg.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.linalg.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.linalg.cross(pos_ji, pos_jref_j)
        plane2 = torch.linalg.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.linalg.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        if self.use_ewald:
            if self.use_pbc:
                lattice = lattice_params_to_matrix_torch(data.lengths, data.angles)
                # Compute reciprocal lattice basis of structure
                k_cell, _ = x_to_k_cell(lattice)
                # Translate lattice indices to k-vectors
                k_grid = torch.matmul(
                    self.k_index_product_set.to(batch.device), k_cell
                )
            else:
                k_grid = (
                    self.k_grid.to(batch.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1)
                )
        # ========== Message Passing Blocks ==========
        for idx, interaction_block in enumerate(self.interaction_blocks):
            #  Step 1: ComENet短程更新
            x_short = interaction_block(x, feature1, feature2, edge_index, batch)
            # Long-range interaction (Ewald, if enabled)
            if self.use_ewald:
                h_ewald, dot, sinc_damping = self.ewald_blocks[idx](
                    h_ewald, pos_detach, k_grid, batch_size, batch, dot, sinc_damping
                )
            else:
                h_ewald = 0

            if self.use_atom_to_atom_mp:
                h_at = self.interactions_at[idx](
                    h_ewald, edge_index_at, edge_weight_at, edge_attr_at
                )
            else:
                h_at = 0
                # Combine short and long range with skip connection
            x = self.skip_connection_factor * (x_short + h_ewald + h_at)
        # ========== Output Layers ==========
        for lin in self.lins:
            x = self.act(lin(x))
        P = self.lin_out(x)

        # Use mean
        if batch is None:
            if self.readout == 'mean':
                energy = P.mean(dim=0)
            elif self.readout == 'sum':
                energy = P.sum(dim=0)
            elif self.readout == 'cat':
                import pdb
                pdb.set_trace()
                energy = torch.cat([P.sum(dim=0), P.mean(dim=0)])
            else:
                raise NotImplementedError
        else:
            # TODO: if want to use cat, need two lines here
            energy = scatter(P, batch, dim=0, reduce=self.readout)


        energy = scatter(x, batch, dim=0)

        return energy

    def forward(self, data):
        return self._forward(data)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
