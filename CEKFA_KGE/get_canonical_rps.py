import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer

from parse_args import *
from data import *


class Contentdataset(Dataset):
    """
    节点或者关系的str
    """
    def __init__(self, args, data_list):
        self.data_list = data_list

        if args.bert_model_dir != None:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
        self.CLS_ID = self.tokenizer.cls_token_id
        self.SEP_ID = self.tokenizer.sep_token_id
        
    def __getitem__(self, index):
        return self.data_list[index]
        
        
    def __len__(self):
        return len(self.data_list)
        
    def getbatch_data(self, raw_batch):           
        nodes_code = self.tokenizer.batch_encode_plus(raw_batch, padding='longest', truncation=True, max_length=100)
        nodes_bert_id = torch.LongTensor(nodes_code["input_ids"])
        return nodes_bert_id
    
    def getbatch_str(self, raw_batch):          
        return raw_batch


def get_sbert_init_embs(args, basedata, content, mth="ENT"):
    if content == "nodes":
        data_list = [basedata.id2ent[i] for i in range(0, basedata.num_nodes)]
        emb_file = os.path.join(args.rpneighs_dir, "Nodes_init_emb_" +args.dataset + "_ambv2.npy")
    elif content == "rels":
        if mth == "ENT": 
            data_list = ['[ENT1] ' + basedata.id2rel[i] + ' [ENT2].' for i in range(0, basedata.num_rels)]
            emb_file = os.path.join(args.rpneighs_dir, "Rels_init_emb_" +args.dataset + "_ambv2_ENT.npy")
        elif mth == "AB": 
            data_list = ['A ' + basedata.id2rel[i] + ' B.' for i in range(0, basedata.num_rels)]
            emb_file = os.path.join(args.rpneighs_dir, "Rels_init_emb_" +args.dataset + "_ambv2_AB.npy")
        elif mth == "raw": 
            data_list = [basedata.id2rel[i] for i in range(0, basedata.num_rels)]
            emb_file = os.path.join(args.rpneighs_dir, "Rels_init_emb_" +args.dataset + "_ambv2_raw.npy")
    else:
        raise Exception

    dataset = Contentdataset(args, data_list)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, collate_fn=dataset.getbatch_str)

    model = SentenceTransformer(args.retrieval_model) 
    
    print('getting file:', emb_file)
    file_name_list = []
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            out = model.encode(batch_data)

            if i > 0 and i % 90 == 0:
                file_name = emb_file[:-4]+str(i)+'.npy'
                file_name_list.append(file_name)
                np.save(file_name, emb)
                print("save part of embeddings in:",file_name)
                emb = out
                continue
            if i == 0:
                emb = out
            else:
                emb =  np.vstack((emb,out))
    file_name = emb_file[:-4]+str(i)+'.npy'
    file_name_list.append(file_name)
    np.save(file_name, emb)
    print("save part of embeddings in :",file_name)
    print('total batch:', i)
    
    for i, file_name in enumerate(file_name_list):
        if i == 0:
            emb = np.load(file_name)
        else:
            embadd = np.load(file_name)
            emb = np.concatenate((emb, embadd), axis=0)
        os.remove(file_name)
        
    np.save(emb_file, emb)
    print("getting initialized embeddings done:", emb_file)

    

def write_canonical_rps(args, fileprefix, rel_scores_all, rel1_info_all, rels2_of1_info_all, relpair_none_line, mth):
    fw = open(os.path.join(args.rpneighs_dir, fileprefix + "_" + mth +"_rp2neighs_scores.txt"), "w")
    fw2 = open(os.path.join(args.rpneighs_dir, fileprefix + "_" + mth + "_rpneighs.txt"), "w")
    for i in range(len(rel_scores_all)):
        rel1_id, rel1_str = rel1_info_all[i]
        rels2_of1_info = rels2_of1_info_all[i]
        rels2_scores = rel_scores_all[i]
        
        rels = []
        for j in range(len(rels2_of1_info)):
            rels.append((rels2_of1_info[j][0], rels2_of1_info[j][1], rels2_scores[j]))
        sortedrels = sorted(rels, key=lambda x:x[2], reverse=True)
        
        rels_str = []
        rels_id_str = []
        for i in range(len(sortedrels)):
            rels_str.append("\t".join([str(sortedrels[i][0]), sortedrels[i][1], str(sortedrels[i][2])]))
            rels_id_str.append(str(sortedrels[i][0]))
        rels_str = ";\t".join(rels_str)   
        rels_id_str = "\t".join(rels_id_str)

        fw.write("%s\t%s:\t%s\n" % (str(rel1_id), rel1_str, rels_str)) 
        fw2.write("%s\t%s\t%s\n" % (str(rel1_id), len(sortedrels), rels_id_str))
    fw.close()
    
    for line in relpair_none_line:
        line = line.strip().split("\t")
        fw2.write("%s\t0\n" % line[0])
    fw2.close()
    return


def get_canonical_rps(args, basedata, fileprefix, mth):
    emb_file = os.path.join(args.rpneighs_dir, "Rels_init_emb_" +args.dataset + "_ambv2_"+mth+".npy")
    if not os.path.exists(emb_file):
        print(emb_file)
        get_sbert_init_embs(args, basedata, "rels", mth)

    rels_emb_withrev = np.load(emb_file)
    rels_emb_withrev = torch.FloatTensor(rels_emb_withrev)
    rels_emb = rels_emb_withrev[:rels_emb_withrev.shape[0]//2]
    num_rels = rels_emb.shape[0]

    rel1_info_all = []
    rels2_of1_info_all = []
    rel_scores_all = []
    for i in range(num_rels):
        query_emb= rels_emb[i]
        cos_scores = util.cos_sim(query_emb, rels_emb)[0]    
        
        cos_scores[i] = -1     
        top_results = torch.topk(cos_scores, k=20)
        rel1_info = [i, basedata.id2rel[i]]
        rels2_of1_info = []
        batch_scores = []
        for score, idx in zip(top_results[0], top_results[1]):
            rels2_of1_info.append([idx.item(), basedata.id2rel[idx.item()]])
            batch_scores.append(score.item())
        
        rel1_info_all.append(rel1_info)
        rels2_of1_info_all.append(rels2_of1_info)
        rel_scores_all.append(batch_scores)
    write_canonical_rps(args, fileprefix, rel_scores_all, rel1_info_all, rels2_of1_info_all, [], mth)
