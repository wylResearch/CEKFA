#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LAN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(LAN, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, Shape = (x.size(0), x.size(0)), x = x)

    def message(self, x_j, edge_index, Shape):
        row, col = edge_index
        deg = degree(row, Shape[0], dtype=x_j.dtype)
        deg_inv = deg.pow(-1.0)

        return deg_inv[row].view(-1, 1) * x_j

    def update(self, h_prime,x):                
        return  x/2 + h_prime/2
    
class KGEModel(nn.Module):
    def __init__(self, args, edges_np, node_id):
        super(KGEModel, self).__init__()
        self.model_name = args.model_name
        model_name = args.model_name
        num_nodes = args.num_nodes
        num_rels = args.num_rels
        hidden_dim = args.hidden_dim
        gamma=args.gamma
        double_entity_embedding=args.double_entity_embedding
        double_relation_embedding=args.double_relation_embedding
        bert_init = args.bert_init

        self.np_neigh_mth = args.np_neigh_mth
        self.edges_np = edges_np
        self.node_id = node_id
        if args.np_neigh_mth == 'LAN':
            self.cn_np = LAN(args.hidden_dim, args.hidden_dim)

        self.rp_neigh_mth = args.rp_neigh_mth

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'PairRE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
        if model_name == 'PairRE' and (not double_relation_embedding):
            raise ValueError('PairRE should use --double_relation_embedding')

        self.epsilon = 2.0
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        self.fc_ent_emb1 = None
        self.fc_ent_emb2 = None
        self.fc_rel_emb1 = None
        self.fc_rel_emb2 = None
        self.dropout = nn.Dropout(args.dp)

        if bert_init:
            nodes_emb = np.load(args.nodes_emb_file)        
            nodes_emb = torch.FloatTensor(nodes_emb)
            nodes_emb_dim = nodes_emb.shape[1] 
            if double_entity_embedding and self.entity_dim == nodes_emb_dim * 2:
                nodes_emb = torch.cat((nodes_emb,nodes_emb),1) 
                if args.fc_bert_init:
                    self.fc_ent_emb1 = nn.Linear(nodes_emb_dim, nodes_emb_dim)
                    self.fc_ent_emb2 = nn.Linear(nodes_emb_dim, nodes_emb_dim)
            self.entity_embedding = nn.Parameter(nodes_emb)
            
            rels_emb = np.load(args.rels_emb_file)
            rels_emb = torch.FloatTensor(rels_emb)
            if self.rp_neigh_mth is None:
                rels_emb = rels_emb[:num_rels,:]
            else:
                rels_emb = rels_emb[:(num_rels-1),:]
                pad_rel_emb = torch.zeros_like(rels_emb[0]).unsqueeze(0)
                rels_emb = torch.cat([rels_emb, pad_rel_emb], 0)
            assert rels_emb.shape[0] == args.num_rels
            rel_emb_dim = rels_emb.shape[1]
            if double_relation_embedding and self.relation_dim == rel_emb_dim * 2:
                rels_emb = torch.cat((rels_emb,rels_emb),1) 
                if args.fc_bert_init:
                    self.fc_rel_emb1 = nn.Linear(rel_emb_dim, rel_emb_dim)
                    self.fc_rel_emb2 = nn.Linear(rel_emb_dim, rel_emb_dim)
            self.relation_embedding = nn.Parameter(rels_emb)
                   
            self.gamma = nn.Parameter(
                torch.Tensor([gamma]), 
                requires_grad=False
            )
            emb_max_1 = torch.max(self.entity_embedding).reshape(-1)
            emb_min_1 = torch.min(self.entity_embedding).reshape(-1)
            emb_max_2 = torch.max(self.relation_embedding).reshape(-1)
            emb_min_2 = torch.min(self.relation_embedding).reshape(-1)
            emb_max = torch.max(torch.abs(torch.cat([emb_max_1, emb_min_1, emb_max_2, emb_min_2]))).reshape(-1)
            self.embedding_range = nn.Parameter(emb_max, requires_grad=False)
        else:
            self.gamma = nn.Parameter(
                torch.Tensor([gamma]), 
                requires_grad=False
            )
            
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
                requires_grad=False
            )
            
            self.entity_embedding = nn.Parameter(torch.zeros(num_nodes, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            
            self.relation_embedding = nn.Parameter(torch.zeros(num_rels, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        

    def forward(self, sample, mode='single', neigh_rels_id=None):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
        
        if self.np_neigh_mth == 'LAN':
            entity_embedding = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=self.node_id
            )
            if self.model_name in ['RotatE', 'ComplEx']:
                ent_1, ent_2 = torch.chunk(entity_embedding, 2, dim=1)
                if self.fc_ent_emb1 is not None and self.fc_ent_emb2 is not None:
                    ent_1 = self.fc_ent_emb1(ent_1)
                    ent_2 = self.fc_ent_emb2(ent_2)
                ent_1 = self.cn_np(ent_1, self.edges_np)   
                ent_2 = self.cn_np(ent_2, self.edges_np)   
                entity_embedding = torch.cat([ent_1,ent_2],1)
            else:
                entity_embedding = self.cn_np(entity_embedding, self.edges_np)   
        else:
            entity_embedding = self.entity_embedding

            
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            # head_part (1024,3)
            # tail_part (1024,128) 
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)

        if neigh_rels_id is not None:
            batch_size, neigh_num = neigh_rels_id.size(0), neigh_rels_id.size(1)
            
            neigh_rels_emb = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=neigh_rels_id.view(-1)
            ).view(batch_size, neigh_num, -1)
        else:
            neigh_rels_emb = None

            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'PairRE': self.PairRE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode, neigh_rels_emb)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    

    def get_avg_rp(self, rel_embed, neigh_embed):
        bs, neigh_num, dim = neigh_embed.shape	
        neigh_embed = neigh_embed.sum(dim=1) / neigh_num
        rel_embed = (rel_embed + neigh_embed.reshape(bs, 1, dim))/2
        return rel_embed

    
    def TransE(self, head, relation, tail, mode, neigh_rels_emb):   
        if neigh_rels_emb is not None:
            relation = self.get_avg_rp(relation, neigh_rels_emb)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail           
            # 正样本时：head:(1024,1,500), relation:(1024,1,500), tail:(1024,1,500), score:(1024,1,500)
            # 负样本的tail-batch时： head:(1024,1,500), relation:(1024,1,500), tail:(1024,128,500), score:(1024,128,500)

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score        # score:(1024,128)
    

    def DistMult(self, head, relation, tail, mode, neigh_rels_emb):
        if neigh_rels_emb is not None:
            relation = self.get_avg_rp(relation, neigh_rels_emb)

        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score


    def ComplEx(self, head, relation, tail, mode, neigh_rels_emb):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        
        if self.np_neigh_mth != 'LAN':
            if self.fc_ent_emb1 is not None and self.fc_ent_emb2 is not None:
                re_head = self.fc_ent_emb1(re_head)
                im_head = self.fc_ent_emb2(im_head)
                re_tail = self.fc_ent_emb1(re_tail)
                im_tail = self.fc_ent_emb2(im_tail)
        
        if self.fc_rel_emb1 is not None and self.fc_rel_emb2 is not None:
            re_relation = self.fc_rel_emb1(re_relation)
            im_relation = self.fc_rel_emb2(im_relation)

        if neigh_rels_emb is not None:
            re_neigh_rels_emb, im_neigh_rels_emb = torch.chunk(neigh_rels_emb, 2, dim=2)
            if self.fc_rel_emb1 is not None and self.fc_rel_emb2 is not None:
                re_neigh_rels_emb = self.fc_rel_emb1(re_neigh_rels_emb)
                im_neigh_rels_emb = self.fc_rel_emb2(im_neigh_rels_emb)
            re_relation = self.get_avg_rp(re_relation, re_neigh_rels_emb)
            im_relation = self.get_avg_rp(im_relation, im_neigh_rels_emb)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score


    def RotatE(self, head, relation, tail, mode, neigh_rels_emb):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        
        if self.np_neigh_mth != 'LAN':
            if self.fc_ent_emb1 is not None and self.fc_ent_emb2 is not None:
                re_head = self.fc_ent_emb1(re_head)
                im_head = self.fc_ent_emb2(im_head)
                re_tail = self.fc_ent_emb1(re_tail)
                im_tail = self.fc_ent_emb2(im_tail)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if neigh_rels_emb is not None:
            phase_neigh_rels_emb = neigh_rels_emb/(self.embedding_range.item()/pi)
            re_neigh_rels_emb = torch.cos(phase_neigh_rels_emb)
            im_neigh_rels_emb = torch.sin(phase_neigh_rels_emb)
            re_relation = self.get_avg_rp(re_relation, re_neigh_rels_emb)
            im_relation = self.get_avg_rp(im_relation, im_neigh_rels_emb)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score


    def pRotatE(self, head, relation, tail, mode, neigh_rels_emb):
        raise NotImplementedError
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if neigh_rels_emb is not None:
            raise NotImplementedError

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    

    def PairRE(self, head, relation, tail, mode, neigh_rels_emb):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)          # wyl   relation:(512,1,200),    re_head, re_tail:(512,1,100)
        if self.fc_rel_emb1 is not None and self.fc_rel_emb2 is not None:
            re_head = self.fc_rel_emb1(re_head)
            re_tail = self.fc_rel_emb2(re_tail)

        if neigh_rels_emb is not None:
            neigh_rels_emb_head, neigh_rels_emb_tail = torch.chunk(neigh_rels_emb, 2, dim=2)
            if self.fc_rel_emb1 is not None and self.fc_rel_emb2 is not None:
                neigh_rels_emb_head = self.fc_rel_emb1(neigh_rels_emb_head)
                neigh_rels_emb_tail = self.fc_rel_emb2(neigh_rels_emb_tail)
            re_head = self.get_avg_rp(re_head, neigh_rels_emb_head)
            re_tail = self.get_avg_rp(re_tail, neigh_rels_emb_tail)

        head = F.normalize(head, 2, -1) 
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail                     # (512,1,100) * (512,1,100) - (512,128,100) * (512,1,100) = (512,128,100)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
    

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode, neigh_rels_id = next(train_iterator)
        # positive_sample (1024,3)
        # negative_sample (1024,128)    # 每个正样本有128个负样本(替换了三元组中的head或者tail)

        positive_sample = positive_sample.to(args.device)
        negative_sample = negative_sample.to(args.device)
        subsampling_weight = subsampling_weight.to(args.device)
        if neigh_rels_id is not None:
            neigh_rels_id = neigh_rels_id.to(args.device)

        negative_score = model((positive_sample, negative_sample), mode=mode, neigh_rels_id=neigh_rels_id)
        # negative_score:(1024,128)
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)
        # negative_score:(1024)

        positive_score = model(positive_sample, neigh_rels_id=neigh_rels_id) # mode 为 single  
        # positive_score:(1024)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        
        #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.num_nodes, 
                args.num_rels, 
                'head-batch',
                args.rps2neighs, 
                args.inverse2rel_map,

            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                args.num_nodes, 
                args.num_rels, 
                'tail-batch',
                args.rps2neighs, 
                args.inverse2rel_map,
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        headflag = True
        tailflag = True
        
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode, neigh_rels_id in test_dataset:
                    
                    positive_sample = positive_sample.to(args.device)
                    negative_sample = negative_sample.to(args.device)
                    filter_bias = filter_bias.to(args.device)
                    if neigh_rels_id is not None:
                        neigh_rels_id = neigh_rels_id.to(args.device)

                    batch_size = positive_sample.size(0)

                    test_scores_batch = model((positive_sample, negative_sample), mode, neigh_rels_id)
                    test_scores_batch += filter_bias

                    ### 
                    if headflag and mode == "head-batch":
                        test_scores_head = test_scores_batch.cpu()
                        headflag = False
                    elif tailflag and mode == "tail-batch":
                        test_scores_tail = test_scores_batch.cpu()   
                        tailflag = False 
                    else:
                        if mode == "head-batch":
                            test_scores_head = torch.cat((test_scores_head,test_scores_batch.cpu()), 0)
                        else:
                            test_scores_tail = torch.cat((test_scores_tail,test_scores_batch.cpu()), 0)


                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(test_scores_batch, dim = 1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero(as_tuple=False)
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        test_scores = [test_scores_head.data.numpy(), test_scores_tail.data.numpy()]
        return metrics, test_scores
