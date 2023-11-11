#### Import all the supporting classes
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cn_variants import LAN, LAN_Separate
from modules import *



class BertResNet_2Inp(nn.Module):
    """
    在 Care_BertResNet 的基础上, 可以考虑 全局或者局部的关系规范化
    卷积的输入是2个input(实体和关系)
    """
    def __init__(self, args, edges_np, node_id):
        super(BertResNet_2Inp, self).__init__()
        self.args = args
        self.edges_np = edges_np
        self.node_id = node_id

        if args.bert_init:
            print(args.nodes_emb_file)
            nodes_emb = np.load(args.nodes_emb_file)        
            nodes_emb = torch.FloatTensor(nodes_emb)
            self.np_embeddings = nn.Embedding.from_pretrained(nodes_emb, freeze=False)

            print(args.rels_emb_file)
            rels_emb = np.load(args.rels_emb_file)
            rels_emb = torch.FloatTensor(rels_emb)
            if rels_emb.shape[0] != args.num_rels:
                pad_rel_emb = torch.zeros_like(rels_emb[0]).unsqueeze(0)
                rels_emb = torch.cat([rels_emb, pad_rel_emb], 0)
                assert rels_emb.shape[0] == args.num_rels
            self.rp_embeddings = nn.Embedding.from_pretrained(rels_emb, freeze=False)
        else:
            self.np_embeddings = nn.Embedding(self.args.num_nodes, self.args.nfeats)        # TODO
            nn.init.xavier_normal_(self.np_embeddings.weight.data)
            self.rp_embeddings = nn.Embedding(self.args.num_rels, self.args.nfeats)        # TODO
            nn.init.xavier_normal_(self.rp_embeddings.weight.data)

        self.np_neigh_mth = args.np_neigh_mth
        if args.np_neigh_mth == 'LAN':
            self.cn_np = LAN(self.args.bert_dim, self.args.bert_dim)
            print("Global canonicalization of np.")
        else:
            print("none canonicalization of np.")

        self.rp_neigh_mth = args.rp_neigh_mth
        if self.rp_neigh_mth == 'Global':
            self.cn_rp = LAN(self.args.bert_dim, self.args.bert_dim)
            print("Global canonicalization of rp.")
        elif self.rp_neigh_mth == 'Local':
            print("Local canonicalization of rp.")
        else:
            print("none canonicalization of rp.")

        self.convhr = ConvEntAndRel(args)
        

    def forward(self, data):
        samples = data["pairs_id"]
        edges_np, node_id = self.edges_np, self.node_id

        if self.np_neigh_mth == 'LAN':
            np_embed = self.np_embeddings(node_id)
            np_embed = self.cn_np(np_embed, edges_np)   
            sub_embed = np_embed[samples[:,0]]
        else:
            np_embed = self.np_embeddings(node_id)
            sub_embed = self.np_embeddings(samples[:,0])

        if self.rp_neigh_mth == 'Global':
            edges_rp, rel_id = data["edges_rp"], data["rel_id"]
            rp_embed = self.rp_embeddings(rel_id)
            rp_embed = self.cn_rp(rp_embed, edges_rp) 
            rel_embed = rp_embed[samples[:,1]]
        elif self.rp_neigh_mth == 'Local': # "Navg" 
            neigh_batch = data["neigh_rels_id"]
            bs = samples.shape[0]
            rel_embed = self.rp_embeddings(samples[:,1])
            neigh_embed = self.rp_embeddings(neigh_batch)
            neigh_embed = neigh_embed.reshape(bs, self.args.maxnum_rpneighs, self.args.bert_dim)	
            neigh_embed = neigh_embed.sum(dim=1) / self.args.maxnum_rpneighs
            rel_embed = (rel_embed + neigh_embed)/2
        else:
            rel_embed = self.rp_embeddings(samples[:,1])

        obj_embed = self.convhr(sub_embed, rel_embed)    
        
        scores = torch.mm(obj_embed, np_embed.transpose(1,0))  
        
        return scores



class BertResNet_3Inp(nn.Module):
    """
    在 CaRE_BertResNet_v4 的基础上, 可以考虑 全局或者局部的关系规范化
    CaRE_BertResNet_v4 是将 实体规范化部分的x 和其邻居的表示分开
    BertResNet 中的输入改为 (bs,3,768) 经过conv1d得到 (bs,reshape_len**2,768)
    """
    def __init__(self, args, edges_np, node_id):
        super(BertResNet_3Inp, self).__init__()
        self.args = args
        self.edges_np = edges_np
        self.node_id = node_id

        if args.bert_init:
            print(args.nodes_emb_file)
            nodes_emb = np.load(args.nodes_emb_file)        
            nodes_emb = torch.FloatTensor(nodes_emb)
            self.np_embeddings = nn.Embedding.from_pretrained(nodes_emb, freeze=False)

            print(args.rels_emb_file)
            rels_emb = np.load(args.rels_emb_file)
            rels_emb = torch.FloatTensor(rels_emb)
            if rels_emb.shape[0] != args.num_rels:
                pad_rel_emb = torch.zeros_like(rels_emb[0]).unsqueeze(0)
                rels_emb = torch.cat([rels_emb, pad_rel_emb], 0)
                assert rels_emb.shape[0] == args.num_rels
            self.rp_embeddings = nn.Embedding.from_pretrained(rels_emb, freeze=False)
        else:
            self.np_embeddings = nn.Embedding(self.args.num_nodes, self.args.nfeats)        # TODO
            nn.init.xavier_normal_(self.np_embeddings.weight.data)
            self.rp_embeddings = nn.Embedding(self.args.num_rels, self.args.nfeats)        # TODO
            nn.init.xavier_normal_(self.rp_embeddings.weight.data)

        self.np_neigh_mth = args.np_neigh_mth
        if args.np_neigh_mth == 'LAN':
            self.cn_np = LAN_Separate(self.args.bert_dim, self.args.bert_dim)
            print("Global canonicalization of np.")
        else:
            raise ValueError

        self.rp_neigh_mth = args.rp_neigh_mth
        if self.rp_neigh_mth == 'Global':
            self.cn_rp = LAN(self.args.bert_dim, self.args.bert_dim)
            print("Global canonicalization of rp.")
        elif self.rp_neigh_mth == 'Local':
            print("Local canonicalization of rp.")
        else:
            print("none canonicalization of rp.")

        self.convhr = ConvEntAndRel(args, input_c=3)

    def forward(self, data):
        samples = data["pairs_id"]
        edges_np, node_id = self.edges_np, self.node_id

        if self.np_neigh_mth == 'LAN':
            np_embed = self.np_embeddings(node_id)
            np_embed, np_canonical = self.cn_np(np_embed, edges_np)   
            sub_embed = np_embed[samples[:,0]]
            sub_canonical = np_canonical[samples[:,0]]

        if self.rp_neigh_mth == 'Global':
            edges_rp, rel_id = data["edges_rp"], data["rel_id"]
            rp_embed = self.rp_embeddings(rel_id)
            rp_embed = self.cn_rp(rp_embed, edges_rp) 
            rel_embed = rp_embed[samples[:,1]]
        elif self.rp_neigh_mth == 'Local': # "Navg" 
            neigh_batch = data["neigh_rels_id"]
            bs = samples.shape[0]
            rel_embed = self.rp_embeddings(samples[:,1])
            neigh_embed = self.rp_embeddings(neigh_batch)
            neigh_embed = neigh_embed.reshape(bs, self.args.maxnum_rpneighs, self.args.bert_dim)	
            neigh_embed = neigh_embed.sum(dim=1) / self.args.maxnum_rpneighs
            rel_embed = (rel_embed + neigh_embed)/2
        else:
            rel_embed = self.rp_embeddings(samples[:,1])
        
        obj_embed = self.convhr(sub_embed, rel_embed, sub_canonical)  

        scores = torch.mm(obj_embed, np_embed.transpose(1,0))  

        return scores



class ConvE(nn.Module):
    def __init__(self, args, edges_np, node_id):
        super(ConvE, self).__init__()
        self.args = args
        self.edges_np = edges_np
        self.node_id = node_id

        if args.bert_init:
            print(args.nodes_emb_file)
            nodes_emb = np.load(args.nodes_emb_file)        
            nodes_emb = torch.FloatTensor(nodes_emb)
            self.np_embeddings = nn.Embedding.from_pretrained(nodes_emb, freeze=False)

            print(args.rels_emb_file)
            rels_emb = np.load(args.rels_emb_file)
            rels_emb = torch.FloatTensor(rels_emb)
            if rels_emb.shape[0] != args.num_rels:
                pad_rel_emb = torch.zeros_like(rels_emb[0]).unsqueeze(0)
                rels_emb = torch.cat([rels_emb, pad_rel_emb], 0)
                assert rels_emb.shape[0] == args.num_rels
            self.rp_embeddings = nn.Embedding.from_pretrained(rels_emb, freeze=False)
        else:
            self.np_embeddings = nn.Embedding(self.args.num_nodes, self.args.nfeats)        # TODO
            nn.init.xavier_normal_(self.np_embeddings.weight.data)
            self.rp_embeddings = nn.Embedding(self.args.num_rels, self.args.nfeats)        # TODO
            nn.init.xavier_normal_(self.rp_embeddings.weight.data)
        
        self.np_neigh_mth = args.np_neigh_mth
        if args.np_neigh_mth == 'LAN':
            self.cn_np = LAN(self.args.bert_dim, self.args.bert_dim)
            print("Global canonicalization of np.")
        else:
            print("none canonicalization of np.")

        self.rp_neigh_mth = args.rp_neigh_mth
        if self.rp_neigh_mth == 'Global':
            self.cn_rp = LAN(self.args.bert_dim, self.args.bert_dim)
            print("Global canonicalization of rp.")
        elif self.rp_neigh_mth == 'Local':
            print("Local canonicalization of rp.")
        else:
            print("none canonicalization of rp.")

        # ConvE
        self.inp_drop = torch.nn.Dropout(self.args.dropout)
        self.hidden_drop = torch.nn.Dropout(self.args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(self.args.dropout)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))     # 输入为(C, )
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.fc = torch.nn.Linear(44160,self.args.bert_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.args.bert_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.args.num_nodes)))


    def get_scores(self, ent, rel, ent_embed):
        batch_size = ent.shape[0]
        ent = ent.view(-1, 1, 24, 32)
        rel = rel.view(-1, 1, 24, 32)

        stacked_inputs = torch.cat([ent, rel], 2)       # (bs, 1, 48, 32)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)                               # (bs, 32, 46, 30)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)                      # (bs,44160)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_embed.transpose(1,0))
        x += self.b.expand_as(x)
        return x    
        

    def forward(self, data):
        samples = data["pairs_id"]
        edges_np, node_id = self.edges_np, self.node_id

        if self.np_neigh_mth == 'LAN':
            np_embed = self.np_embeddings(node_id)
            np_embed = self.cn_np(np_embed, edges_np)   
            sub_embed = np_embed[samples[:,0]]
        else:
            np_embed = self.np_embeddings(node_id)
            sub_embed = self.np_embeddings(samples[:,0])
        
        if self.rp_neigh_mth == 'Global':
            edges_rp, rel_id = data["edges_rp"], data["rel_id"]
            rp_embed = self.rp_embeddings(rel_id)
            rp_embed = self.cn_rp(rp_embed, edges_rp) 
            rel_embed = rp_embed[samples[:,1]]
        elif self.rp_neigh_mth == 'Local': # "Navg" 
            neigh_batch = data["neigh_rels_id"]
            bs = samples.shape[0]
            rel_embed = self.rp_embeddings(samples[:,1])
            neigh_embed = self.rp_embeddings(neigh_batch)
            neigh_embed = neigh_embed.reshape(bs, self.args.maxnum_rpneighs, self.args.bert_dim)	
            neigh_embed = neigh_embed.sum(dim=1) / self.args.maxnum_rpneighs
            rel_embed = (rel_embed + neigh_embed)/2
        else:
            rel_embed = self.rp_embeddings(samples[:,1])

        scores = self.get_scores(sub_embed, rel_embed, np_embed) 
        
        return scores


