import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np




class FeatEmbedding(torch.nn.Module):
    def __init__(self, args):
        super(FeatEmbedding, self).__init__()

        self.geopotential_surface = np.load("/root/autodl-tmp/geopotential_surface.npz")["arr_0"]
        self.land_sea_mask = np.load("/root/autodl-tmp/land_sea_mask.npz")["arr_0"]
        self.geopotential_surface = torch.tensor(self.geopotential_surface).reshape(1,161*161,1).repeat(args.batch_size, 1, 1).to("cuda")
        self.land_sea_mask = torch.tensor(self.land_sea_mask).reshape(1,161*161,1).repeat(args.batch_size, 1, 1).to("cuda")

        gdim, mdim, edim = args.grid_node_dim, args.mesh_node_dim, args.edge_dim
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim
        # 70*2+2, 3, 4
        # 256, 128, 128

        # Embedding the input features
        self.grid_feat_embedding = nn.Sequential(
            nn.Linear(gdim, gemb, bias=True)
        )
        self.mesh_feat_embedding = nn.Sequential(
            nn.Linear(mdim, memb, bias=True)
        )
        self.mesh_edge_feat_embedding = nn.Sequential(
            nn.Linear(edim, eemb, bias=True)
        )
        self.grid2mesh_edge_feat_embedding = nn.Sequential(
            nn.Linear(edim, eemb, bias=True)
        )
        self.mesh2grid_edge_feat_embedding = nn.Sequential(
            nn.Linear(edim, eemb, bias=True)
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):
        """
        gx: [bs,t,c,h,w]  [1, 2, 70, 161, 161]
        mx: [mesh_node*mesh_node, 3]
        me_i: [2, edge_num]  [2, 4722]  mesh edge index
        me_x: [edge_num, 4]  [4722, 4]
        g2me_i: [2, grid2mesh_edge_num]  [2, 36386]  grid2mesh edge index
        g2me_x: [grid2mesh_edge_num, 4]  [36386, 4]
        m2ge_i: [2, mesh2grid_edge_num]  [2, 56672]  mesh2grid edge index
        m2ge_x: [mesh2grid_edge_num, 4]  [56672, 4]
        """
        bs,t,c,h,w = gx.shape
        gx = gx.reshape(bs, t*c, h*w).transpose(1,2)  # 维度为[b, h*w, t*c]

        gx = torch.concat([gx, self.geopotential_surface, self.land_sea_mask], dim=-1)
        gx = self.grid_feat_embedding(gx)

        mx = self.mesh_feat_embedding(mx).repeat(bs, 1, 1)  # repeat就是将后两个维度不变，第一个为维度重复bs次
        me_x = self.mesh_edge_feat_embedding(me_x).repeat(bs, 1, 1)
        g2me_x = self.grid2mesh_edge_feat_embedding(g2me_x).repeat(bs, 1, 1)
        m2ge_x = self.mesh2grid_edge_feat_embedding(m2ge_x).repeat(bs, 1, 1)

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Grid2MeshEdgeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Grid2MeshEdgeUpdate, self).__init__()

        g2m_enum = args.grid2mesh_edge_num
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim
        # 36386
        # 256, 128, 128

        # Grid2Mesh GNN
        # grid, mesh nodes和本身来更新edges
        self.grid2mesh_edge_update = nn.Sequential(
            nn.Linear(gemb + memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, eemb, bias=True),
            nn.LayerNorm([g2m_enum, eemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):
        """
        gx: [bs,h*w, gemb]
        mx: [mesh_node*mesh_node, memb]
        me_i: [2, edge_num]
        me_x: [edge_num, eemb]
        g2me_i: [2, grid2mesh_edge_num]
        g2me_x: [grid2mesh_edge_num, eemb]
        m2ge_i: [2, mesh2grid_edge_num]
        m2ge_x: [mesh2grid_edge_num, eemb]
        """
        row, col = g2me_i

        # edge update
        # [bs, grid2mesh_edge_num, gemb], [bs, grid2mesh_edge_num, memb], [bs, grid2mesh_edge_num, eemb]
        edge_attr_updated = torch.cat([gx[:, row], mx[:, col], g2me_x], dim=-1)
        edge_attr_updated = self.grid2mesh_edge_update(edge_attr_updated)

        # residual
        g2me_x = g2me_x + edge_attr_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Grid2MeshNodeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Grid2MeshNodeUpdate, self).__init__()

        gnum, mnum = args.grid_node_num, args.mesh_node_num
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim

        # Grid2Mesh GNN
        self.grid2mesh_node_aggregate = nn.Sequential(
            nn.Linear(memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, memb, bias=True),
            nn.LayerNorm([mnum, memb])
        )
        self.grid2mesh_grid_update = nn.Sequential(
            nn.Linear(gemb, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, gemb, bias=True),
            nn.LayerNorm([gnum, gemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):
        """
        gx: [bs,h*w, gemb]
        mx: [mesh_node*mesh_node, memb]
        me_i: [2, edge_num]
        me_x: [edge_num, eemb]
        g2me_i: [2, grid2mesh_edge_num]
        g2me_x: [grid2mesh_edge_num, eemb]
        m2ge_i: [2, mesh2grid_edge_num]
        m2ge_x: [mesh2grid_edge_num, eemb]
        """
        row, col = g2me_i

        # mesh node update
        edge_agg = scatter(g2me_x, col, dim=-2, reduce='sum')  # 聚合mesh中结点，所以是col
        mesh_node_updated = torch.cat([mx, edge_agg], dim=-1)
        mesh_node_updated = self.grid2mesh_node_aggregate(mesh_node_updated)

        # grid node update
        grid_node_updated = self.grid2mesh_grid_update(gx)

        # residual
        gx = gx + grid_node_updated
        mx = mx + mesh_node_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class MeshEdgeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(MeshEdgeUpdate, self).__init__()

        m_enum = args.mesh_edge_num
        memb, eemb = args.mesh_node_embed_dim, args.edge_embed_dim

        # Multi-mesh GNN
        self.mesh_edge_update = nn.Sequential(
            nn.Linear(memb + memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, eemb, bias=True),
            nn.LayerNorm([m_enum, eemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = me_i

        # edge update
        edge_attr_updated = torch.cat([mx[:, row], mx[:, col], me_x], dim=-1)
        edge_attr_updated = self.mesh_edge_update(edge_attr_updated)

        # residual
        me_x = me_x + edge_attr_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class MeshNodeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(MeshNodeUpdate, self).__init__()

        mnum = args.mesh_node_num
        memb, eemb = args.mesh_node_embed_dim, args.edge_embed_dim

        # Grid2Mesh GNN
        self.mesh_node_aggregate = nn.Sequential(
            nn.Linear(memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, memb, bias=True),
            nn.LayerNorm([mnum, memb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = me_i

        # mesh node update
        edge_agg = scatter(me_x, col, dim=-2, reduce='sum')
        mesh_node_updated = torch.cat([mx, edge_agg], dim=-1)
        mesh_node_updated = self.mesh_node_aggregate(mesh_node_updated)

        # residual
        mx = mx + mesh_node_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Mesh2GridEdgeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Mesh2GridEdgeUpdate, self).__init__()

        m2g_enum = args.mesh2grid_edge_num
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim

        # Mesh2grid GNN
        self.mesh2grid_edge_update = nn.Sequential(
            nn.Linear(gemb + memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, eemb, bias=True),
            nn.LayerNorm([m2g_enum, eemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = m2ge_i

        # edge update
        edge_attr_updated = torch.cat([mx[:, row], gx[:, col], m2ge_x], dim=-1)
        edge_attr_updated = self.mesh2grid_edge_update(edge_attr_updated)

        # residual
        m2ge_x = m2ge_x + edge_attr_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Mesh2GridNodeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Mesh2GridNodeUpdate, self).__init__()

        gnum = args.grid_node_num
        gemb, eemb = args.grid_node_embed_dim, args.edge_embed_dim

        # Mesh2grid GNN
        self.mesh2grid_node_aggregate = nn.Sequential(
            nn.Linear(gemb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, gemb, bias=True),
            nn.LayerNorm([gnum, gemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = m2ge_i

        # grid node update
        edge_agg = scatter(m2ge_x, col, dim=-2, reduce='sum')
        grid_node_updated = torch.cat([gx, edge_agg], dim=-1)
        grid_node_updated = self.mesh2grid_node_aggregate(grid_node_updated)

        # residual
        gx = gx + grid_node_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class PredictNet(torch.nn.Module):
    def __init__(self, args):
        super(PredictNet, self).__init__()

        gemb = args.grid_node_embed_dim
        pred_dim = args.grid_node_pred_dim

        # prediction
        self.predict_nn = nn.Sequential(
            nn.Linear(gemb, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, 128, bias=True),
            nn.SiLU(),
            nn.Linear(128, pred_dim, bias=True)
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):
        # output
        gx = self.predict_nn(gx)

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x

class GraphCast(torch.nn.Module):
    def __init__(self, args):   
        super(GraphCast, self).__init__()

        embed = FeatEmbedding(args)
        gnn_blocks = [
            Grid2MeshEdgeUpdate(args),
            Grid2MeshNodeUpdate(args),
            MeshEdgeUpdate(args),
            MeshNodeUpdate(args),
            MeshEdgeUpdate(args),
            MeshNodeUpdate(args),
            Mesh2GridEdgeUpdate(args),
            Mesh2GridNodeUpdate(args),
        ]
        head = PredictNet(args)
        layers = [embed] + gnn_blocks + [head]
        self.ll = nn.ModuleList(layers)  # 使用*layer，就是把layer里面的内容取出来用，而不是传递的list

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):
        for ll in self.ll:
            gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x = ll(gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x)
        return gx