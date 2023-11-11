import os
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import logging


class load_basedata():
    def __init__(self, args):
        self.args = args
        self.data_files = self.args.data_files
        self.add_reverse = args.reverse
        self.inverse2rel_map = {}

        self.fetch_data()
        
        self.num_nodes = len(self.ent2id)
        self.num_rels = len(self.rel2id)
    

    def read_canonical_rps(self):
        if self.args.rp_neigh_mth == 'Local':
            if "rp2neighs_scores" in self.data_files["rpneighs_filepath"]:
                self.rps2neighs = self.get_rps2neighs_score(self.data_files["rpneighs_filepath"])
            elif "rpneighs" in self.data_files["rpneighs_filepath"]:
                self.rps2neighs = self.get_rps2neighs(self.data_files["rpneighs_filepath"])
            else:
                self.rps2neighs = None
        else:
            self.rps2neighs = None


    def fetch_data(self):
        self.rel2id,self.id2rel = self.get_relation_phrases()
        self.ent2id,self.id2ent = self.get_noun_phrases()

        self.canon_clusts, self.entid2clustid_cesi, self.unique_clusts, self.num_groups_cesi_np = self.get_clusters(self.data_files["cesi_npclust_path"])
        self.true_clusts, self.entid2clustid, _ , self.num_groups_gold_np = self.get_clusters(self.data_files["gold_npclust_path"])

        self.edges = self.get_edges_np(self.canon_clusts)    # 更新 self.edges, shape为 (2, X) 其中X为规范化边的数目

        self.train_trips,self.rel2id,self.label_graph, self.train_trips_without_rev, label_graph_other = self.get_train_triples(
                                                                                            self.data_files["train_trip_path"],
                                                                                            self.entid2clustid,self.rel2id,
                                                                                            self.id2rel)
        self.label_graph_rt2h, self.label_graph_ht2r, self.label_graph_r2ht, self.label_graph_tr2h = label_graph_other

        self.test_trips, self.test_trips_without_rev = self.get_test_triples(self.data_files["test_trip_path"],self.rel2id, self.id2rel)
        self.valid_trips, self.valid_trips_without_rev = self.get_test_triples(self.data_files["valid_trip_path"],self.rel2id, self.id2rel)


    def get_relation_phrases(self):
        f = open(self.data_files["rel2id_path"],"r").readlines()
        phrase2id = {}
        id2phrase = {}
        phrases = []
        for line in f[1:]:
            line = line.strip().split("\t")
            phrase,ID = line[0], line[1]
            phrases.append(phrase)
            phrase2id[phrase] = int(ID)
            id2phrase[int(ID)] = phrase
        cur_id = max(id2phrase.keys())
        if self.add_reverse:
            for phrase in phrases:
                newphrase =  "inverse of " + phrase
                cur_id += 1 
                phrase2id[newphrase] = cur_id                       # 添加反关系
                id2phrase[cur_id] = newphrase
                self.inverse2rel_map[cur_id] = phrase2id[phrase]    # 反关系id 映射到 关系id
                self.inverse2rel_map[phrase2id[phrase]] = cur_id    # 关系id 映射到 反关系id    
        return phrase2id,id2phrase

    def get_noun_phrases(self):
        f = open(self.data_files["ent2id_path"],"r").readlines()
        phrase2id = {}
        id2phrase = {}
        for line in f[1:]:
            line = line.strip().split("\t")
            phrase,ID = line[0], line[1]
            phrase2id[phrase] = int(ID)
            id2phrase[int(ID)] = phrase
        return phrase2id,id2phrase

    def get_clusters(self, clust_path):
        content_clusts = {}
        contentid2clustid = {}
        content_list = []
        unique_clusts = []           
        ID = -1                      # wyl
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split()
            clust = [int(content) for content in line[2:]]  
            content_clusts[int(line[0])] = clust
            if line[0] not in content_list:
                ID+=1                                       # 簇的 id
                unique_clusts.append(clust)                 # 存储 簇信息
                content_list.extend(line[2:])
                for content in clust: contentid2clustid[content] = ID
        return content_clusts, contentid2clustid, unique_clusts, ID+1
    
    def get_edges_np(self, content_clusts):     # 规范化边
        head_list = []
        tail_list = []
        for content in content_clusts:
            if len(content_clusts[content])==1: # 簇里只有自己，则加上自己
                head_list.append(content)
                tail_list.append(content)
            for _, neigh in enumerate(content_clusts[content]):           # 簇里还有邻居，则不加自己
                if neigh!=content:           
                    head_list.append(neigh)
                    tail_list.append(content)   

        head_list = np.array(head_list).reshape(1,-1)
        tail_list = np.array(tail_list).reshape(1,-1)

        edges = np.concatenate((np.array(head_list),np.array(tail_list)),axis = 0)
        return edges
    
    def get_rps2neighs(self, clust_path):
        self.rel2id["PADRP"] = self.num_rels
        self.id2rel[self.num_rels] = "PADRP"
        pad_neigh_id = self.num_rels
        self.inverse2rel_map[pad_neigh_id] = pad_neigh_id
        self.num_rels += 1
        rps2neighs = {}
        f = open(clust_path,"r").readlines()
        for line in f:
            line = line.strip().split()
            content, neigh_num = int(line[0]), int(line[1])
            neighs = []
            if neigh_num != 0:
                clust = [int(content) for content in line[2:]]  
                neighs = clust[:min(len(clust), self.args.maxnum_rpneighs)]
            if len(neighs) < self.args.maxnum_rpneighs:
                neighs += [pad_neigh_id] * (self.args.maxnum_rpneighs - len(neighs))
            rps2neighs[content] = neighs
        return rps2neighs
    
    def get_rps2neighs_score(self, clust_path):
        self.rel2id["PADRP"] = self.num_rels
        self.id2rel[self.num_rels] = "PADRP"
        pad_neigh_id = self.num_rels
        self.inverse2rel_map[pad_neigh_id] = pad_neigh_id
        self.num_rels += 1
        rps2neighs = {}
        f = open(clust_path,"r").readlines()
        for line_ in f:
            line = line_.strip().split(":")       
            if len(line)!=2:
                line_ = line_.replace("n :o", "n o")
                line = line_.strip().split(":")
            if len(line)!=2:
                print('---',line_)
                raise Exception
            content = int(line[0].split("\t")[0])
            neighs = line[-1].split(";")
            neigh_num = len(neighs)

            rp_neighs = []
            if neigh_num != 0:
                for i, neigh in enumerate(neighs): 
                    _, neigh_id, neigh_str, neigh_score = neigh.split("\t")
                    if float(neigh_score) < self.args.thresholds_rp:            # 设置分数阈值 默认0.8
                        break
                    elif len(rp_neighs)==self.args.maxnum_rpneighs:            
                        break
                    else:
                        rp_neighs.append(int(neigh_id))
            if len(rp_neighs) < self.args.maxnum_rpneighs:
                rp_neighs += [pad_neigh_id] * (self.args.maxnum_rpneighs - len(rp_neighs))
            rps2neighs[content] = rp_neighs
        return rps2neighs

    def get_train_triples(self,triples_path,entid2clustid,rel2id,id2rel):
        logging.info("reading file: %s"%triples_path)
        trip_list = []
        trip_list_without_rev = []
        label_graph = {}
        label_graph_rt2h = {}
        label_graph_ht2r = {}
        label_graph_r2ht = {}
        label_graph_tr2h = {}
        self.label_filter = {}   # hr2t
        self.label_filter_tr2h = {}
        rel_counter = []
        f = open(triples_path,"r").readlines()
        for trip in f[1:]:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            rel_counter.append(r)
            if self.add_reverse:
                r_inv = "inverse of " + id2rel[r]       # 添加 反关系
                if r_inv not in rel2id:
                    import pdb; pdb.set_trace()
                    ID = len(rel2id)
                    rel2id[r_inv] = ID

                r_inv = rel2id[r_inv]                   # 反关系的 id
                rel_counter.append(r_inv)

            if (e1,r) not in label_graph:
                label_graph[(e1,r)] = set()
            label_graph[(e1,r)].add(e2)

            if (r,e2) not in label_graph_rt2h:
                label_graph_rt2h[(r,e2)] = set()
            label_graph_rt2h[(r,e2)].add(e1)

            if (e1,e2) not in label_graph_ht2r:
                label_graph_ht2r[(e1,e2)] = set()
            label_graph_ht2r[(e1,e2)].add(r)

            if r not in label_graph_r2ht:
                label_graph_r2ht[(r)] = set()
            label_graph_r2ht[r].add((e1,e2))

            if (e2,r) not in label_graph_tr2h:
                label_graph_tr2h[(e2,r)] = set()
            label_graph_tr2h[(e2,r)].add(e1)

            if self.add_reverse:
                if (e2,r_inv) not in label_graph:
                    label_graph[(e2,r_inv)] = set()
                label_graph[(e2,r_inv)].add(e1)

                if (r_inv,e1) not in label_graph_rt2h:
                    label_graph_rt2h[(r_inv,e1)] = set()
                label_graph_rt2h[(r_inv,e1)].add(e2)

                if (e2,e1) not in label_graph_ht2r:
                    label_graph_ht2r[(e2,e1)] = set()
                label_graph_ht2r[(e2,e1)].add(r_inv)

                if r_inv not in label_graph_r2ht:
                    label_graph_r2ht[(r_inv)] = set()
                label_graph_r2ht[r_inv].add((e2,e1))

            if (e1,r) not in self.label_filter:
                self.label_filter[(e1,r)] = set()
            self.label_filter[(e1,r)].add(entid2clustid[e2])

            if (e2,r) not in self.label_filter_tr2h:
                self.label_filter_tr2h[(e2,r)] = set()
            self.label_filter_tr2h[(e2,r)].add(entid2clustid[e1])

            if self.add_reverse:
                if (e2,r_inv) not in self.label_filter:
                    self.label_filter[(e2,r_inv)] = set()
                self.label_filter[(e2,r_inv)].add(entid2clustid[e1])

            trip_list.append([e1,r,e2])
            trip_list_without_rev.append([e1,r,e2])
            if self.add_reverse:
                trip_list.append([e2,r_inv,e1])
        
        self.rel_counter = Counter(rel_counter)
        return np.array(trip_list),rel2id,label_graph, trip_list_without_rev, [label_graph_rt2h, label_graph_ht2r, label_graph_r2ht, label_graph_tr2h]

    def get_test_triples(self, triples_path,rel2id,id2rel):
        logging.info("reading file: %s"%triples_path)
        trip_list = []
        trip_list_without_rev = []
        f = open(triples_path,"r").readlines()
        for trip in f[1:]:
            trip = trip.strip().split()
            e1,r,e2 = int(trip[0]),int(trip[1]),int(trip[2])
            trip_list.append([e1,r,e2])
            trip_list_without_rev.append([e1,r,e2])
            if self.add_reverse:
                r_inv = "inverse of " + id2rel[r]
                if r_inv not in rel2id:
                    import pdb; pdb.set_trace()
                    ID = len(rel2id)
                    rel2id[r_inv] = ID

                trip_list.append([e2,rel2id[r_inv],e1])
        return np.array(trip_list), trip_list_without_rev
   




    
    ######################################################################################################################
    ######################################################################################################################
      
class DatasetFromScores_Retrieval_ReRank_KGE(Dataset):
    """ only for test data """
    def __init__(self, args, basedata, scores, trips, data1=None):
        if data1 != None:   # v1 或者 v2 类型的retrieval数据
            self.retrieval_data = data1 
            assert scores.shape[0] == trips.shape[0] #== len(self.retrieval_data)
        else:
            self.retrieval_data = None
        
        self.args = args
        self.basedata = basedata
        self.num_nodes = basedata.num_nodes
        self.trips = trips
        self.scores = scores
        self.sep_id_rp = len(basedata.rel2id) / 2 if args.reverse else len(basedata.rel2id)
        self.SEP = "[SEP]"

    
    def __getitem__(self, index):
        if self.retrieval_data != None:
            return self.trips[index], self.scores[index], self.retrieval_data[index], index
        else:
            raise Exception

        
    def __len__(self):
        return len(self.trips)
    
    def get_batch_retrival_data(self, rawbatch):      
        for i in range(len(rawbatch)):   
            test_trips, test_scores, retrieval_data, index = rawbatch[i]
                   
            test_scores_sorted, test_sorts_indices = torch.sort(torch.tensor(test_scores), dim=0, descending=True)
            candi_TopK = test_sorts_indices[:self.args.rerank_Top_K]
            
            rerank_input =[]
            if index%2==0:
                h, r, t = test_trips[0], test_trips[1], test_trips[2]
                for candi_t in candi_TopK.tolist():
                    rerank_input.append([self.basedata.id2ent[h] + ' ' + self.basedata.id2rel[r] + ' '+ self.basedata.id2ent[candi_t], retrieval_data[2]])
            else:
                t, r, h = test_trips[0], test_trips[1], test_trips[2]
                for candi_h in candi_TopK.tolist():       
                    rerank_input.append([self.basedata.id2ent[candi_h] + ' ' + self.basedata.id2rel[r] + ' '+ self.basedata.id2ent[t], retrieval_data[2]])
                
    
        hr = torch.LongTensor([h,r]).unsqueeze(0).repeat(self.args.rerank_Top_K,1)
        t = candi_TopK.long().unsqueeze(1)
        triples_id = torch.cat([hr,t],dim=1)
        return {'triples_id':triples_id, 
                'test_scores_sorted': test_scores_sorted,
                'test_sorts_indices':test_sorts_indices,   
                'rerank_input':rerank_input,   
                }
    








