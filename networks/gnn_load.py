import pickle

import numpy as np
import torch
from torch_geometric.data import Data
import graph_tools as gg


class EarthGraph(object):
    def __init__(self):
        self.mesh_data = None
        self.grid2mesh_data = None
        self.mesh2grid_data = None

    def generate_graph(self):
        mesh_nodes = gg.fetch_mesh_nodes()

        mesh_6_edges, mesh_6_edges_attrs = gg.fetch_mesh_edges(6)
        mesh_5_edges, mesh_5_edges_attrs = gg.fetch_mesh_edges(5)
        mesh_4_edges, mesh_4_edges_attrs = gg.fetch_mesh_edges(4)
        mesh_3_edges, mesh_3_edges_attrs = gg.fetch_mesh_edges(3)
        mesh_2_edges, mesh_2_edges_attrs = gg.fetch_mesh_edges(2)
        mesh_1_edges, mesh_1_edges_attrs = gg.fetch_mesh_edges(1)
        mesh_0_edges, mesh_0_edges_attrs = gg.fetch_mesh_edges(0)

        mesh_edges = mesh_6_edges + mesh_5_edges + mesh_4_edges + mesh_3_edges + mesh_2_edges + mesh_1_edges + mesh_0_edges
        mesh_edges_attrs = mesh_6_edges_attrs + mesh_5_edges_attrs + mesh_4_edges_attrs + mesh_3_edges_attrs + mesh_2_edges_attrs + mesh_1_edges_attrs + mesh_0_edges_attrs

        self.mesh_data = Data(x=torch.tensor(mesh_nodes, dtype=torch.float),
                              edge_index=torch.tensor(mesh_edges, dtype=torch.long).T.contiguous(),
                              edge_attr=torch.tensor(mesh_edges_attrs, dtype=torch.float))

        grid2mesh_edges, grid2mesh_edge_attrs = gg.fetch_grid2mesh_edges()
        self.grid2mesh_data = Data(x=None,
                                   edge_index=torch.tensor(grid2mesh_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(grid2mesh_edge_attrs, dtype=torch.float))

        mesh2grid_edges, mesh2grid_edge_attrs = gg.fetch_mesh2grid_edges()
        self.mesh2grid_data = Data(x=None,
                                   edge_index=torch.tensor(mesh2grid_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(mesh2grid_edge_attrs, dtype=torch.float))


if __name__ == '__main__':
    graph = EarthGraph()
    graph.generate_graph()

    print(graph.mesh_data.x.shape)