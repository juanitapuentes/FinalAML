import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from model.model import DeeperGCN
from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer, MLP, MM_AtomEncoder
import dgl
import dgllife
import numpy as np


from model.model_encoder import AtomEncoder, BondEncoder

import logging


class Transformer(torch.nn.Module):
    def __init__(self, args, is_prot=False, saliency=False):
        super(Transformer, self).__init__()

        # Set PM configuration
        if is_prot:
            self.num_layers = args.num_layers_prot
            mlp_layers = args.mlp_layers_prot
            hidden_channels = args.hidden_channels_prot
            self.msg_norm = args.msg_norm_prot
            learn_msg_scale = args.learn_msg_scale_prot
            self.conv_encode_edge = args.conv_encode_edge_prot

        # Set LM configuration
        else:
            self.molecule_gcn = DeeperGCN(args)
            self.num_layers = args.num_layers
            mlp_layers = args.mlp_layers
            hidden_channels = args.hidden_channels
            self.msg_norm = args.msg_norm
            learn_msg_scale = args.learn_msg_scale
            self.conv_encode_edge = args.conv_encode_edge

        # Set overall model configuration
        self.dropout = args.dropout
        self.block = args.block
        self.add_virtual_node = args.add_virtual_node
        self.training = True
        self.args = args

        num_classes = args.nclasses
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.classification = nn.Linear(hidden_channels, num_classes)
        norm = args.norm

        graph_pooling = args.graph_pooling

        # Print model parameters
        print(
            "The number of layers {}".format(self.num_layers),
            "Aggr aggregation method {}".format(aggr),
            "block: {}".format(self.block),
        )
        if self.block == "res+":
            print("LN/BN->ReLU->GraphConv->Res")
        elif self.block == "res":
            print("GraphConv->LN/BN->ReLU->Res")
        elif self.block == "dense":
            raise NotImplementedError("To be implemented")
        elif self.block == "plain":
            print("GraphConv->LN/BN->ReLU")
        else:
            raise Exception("Unknown block Type")

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3, norm=norm))

        # Set GCN layer configuration
        for layer in range(self.num_layers):
            if conv == "gen":
                gcn = GENConv(
                    hidden_channels,
                    hidden_channels,
                    args,
                    aggr=aggr,
                    t=t,
                    learn_t=self.learn_t,
                    p=p,
                    learn_p=self.learn_p,
                    msg_norm=self.msg_norm,
                    learn_msg_scale=learn_msg_scale,
                    encode_edge=self.conv_encode_edge,
                    bond_encoder=True,
                    norm=norm,
                    mlp_layers=mlp_layers,
                )
            else:
                raise Exception("Unknown Conv Type")
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        # Set embbeding layers
        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if saliency:
            self.atom_encoder = MM_AtomEncoder(emb_dim=hidden_channels)
        else:
            self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        # Set type of pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception("Unknown Pool Type")

        device = torch.device("cuda:" + str(0))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=1, batch_first=True).to(device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6).to(device)

        self.conv1 = torch.nn.Conv1d(128, 64, kernel_size=1).to(device)
        self.conv2 = torch.nn.Conv1d(64,32, kernel_size=1).to(device)
        self.conv3 = torch.nn.Conv1d(32, 1, kernel_size=1).to(device)

        # Set classification layer
        self.graph_pred_linear = torch.nn.Linear(hidden_channels, num_classes)



    def forward(self, input_batch, dropout=True, embeddings=False):

        #breakpoint()
        '''x = input_batch.x
        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        h = self.atom_encoder(x)

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1)
                .to(edge_index.dtype)
                .to(edge_index.device)
            )
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == "res+":

            h = self.gcns[0](h, edge_index, edge_emb) #mismo shape de h (m,128)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                if dropout:
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = (
                        global_add_pool(h2, batch) + virtualnode_embedding
                    )
                    if dropout:
                        virtualnode_embedding = F.dropout(
                            self.mlp_virtualnode_list[layer - 1](
                                virtualnode_embedding_temp
                            ),
                            self.dropout,
                            training=self.training,
                        )

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h
            #breakpoint()
            h = self.norms[self.num_layers - 1](h)
            if dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "res":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "dense":
            raise NotImplementedError("To be implemented")

        elif self.block == "plain":

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception("Unknown block Type")

        device = torch.device("cuda:" + str(0))
        #encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8).to(device)
        #transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)

        #out = transformer_encoder(h_i).to(device)

        #breakpoint()'''
        '''graph_list = []
        for ind in range(1, input_batch.ptr.shape[0]):
            mol = input_batch[ind-1].mol
            graph_list.append(dgllife.utils.mol_to_complete_graph(mol).to(device))

        #breakpoint()
        graph_batch = dgl.batch(graph_list)
        gap = dgl.nn.pytorch.glob.GlobalAttentionPooling(torch.nn.Linear(128, 1)).to(device)
        h_attn = gap(graph_batch, h)'''

        #max = np.max(np.diff(input_batch.ptr.cpu()))
        max = 128
        h_prev = self.molecule_gcn(input_batch)

        pad_list = []
        mask_list = []
        '''for ind in range(1, input_batch.ptr.shape[0]):
            mol = input_batch[ind - 1].mol
            mol_list.append(dgllife.utils.mol_to_complete_graph(mol).to(device))'''

        '''h_graph = self.pool(h_prev, input_batch.batch)

        if self.args.use_prot or embeddings:
            return h_graph
        else:
            return self.graph_pred_linear(h_graph)
            #return self.graph_pred_linear(h_attn)'''

        for ind in range(1, input_batch.ptr.shape[0]):
            #breakpoint()
            atoms = (input_batch.ptr[ind-1], input_batch.ptr[ind])
            h_ = h_prev[atoms[0]:atoms[1]]
            padded = torch.nn.functional.pad(h_, (0, 0, 0, max - h_.shape[0]))
            pad_list.append(padded)
            ones = torch.ones(h_.shape)
            ones = torch.cat((ones, torch.zeros((128 - h_.shape[0], 128))))
            mask_list.append(ones)

        #breakpoint()#
        device = torch.device("cuda:" + str(0))
        padded_batch = torch.stack(pad_list, 0).to(device)
        mask_batch = torch.stack(mask_list, 0).to(device)

        h_trans = self.transformer_encoder(padded_batch, mask_batch)
        h_trans = h_trans.view(h_trans.shape[0], h_trans.shape[2], h_trans.shape[1])

        #breakpoint()
        conv = self.conv1(h_trans)
        conv = self.conv2(conv)
        conv = self.conv3(conv)

        conv = conv.view(1, conv.shape[0], conv.shape[2])[0]
        pred = self.graph_pred_linear(conv)

        return pred

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print("Final t {}".format(ts))
            else:
                logging.info("Epoch {}, t {}".format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print("Final p {}".format(ps))
            else:
                logging.info("Epoch {}, p {}".format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print("Final s {}".format(ss))
            else:
                logging.info("Epoch {}, s {}".format(epoch, ss))
