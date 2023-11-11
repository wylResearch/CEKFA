import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from data import *
from parse_args import *

"""
不进行finetune,直接使用BertModel编码 node/relation 的mention 
最后取mention的tokens的表示的平均作为 node/relation 的初始化嵌入
"""

class NodesRels_dataset(Dataset):
    """
    节点或者关系的str
    """
    def __init__(self, args, basedata, content):
        self.args = args
        self.basedata = basedata
        self.content = content
        self.num_nodes = basedata.num_nodes
        self.num_rels = len(basedata.id2rel)
        self.nodes_bert_id = list(range(0, self.num_nodes))
        self.rels_bert_id = list(range(0, self.num_rels))

        if args.models in ['bert']:
            if args.bert_model_dir != None:
                self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
        else:
            raise NotImplementedError
        
        self.CLS_ID = self.tokenizer.cls_token_id
        self.SEP_ID = self.tokenizer.sep_token_id
        
    def __getitem__(self, index):
        if self.content == "nodes":
            return self.nodes_bert_id[index]
        elif self.content == "rels":
            return self.rels_bert_id[index]
        
        
    def __len__(self):
        if self.content == "nodes":
            return self.num_nodes
        elif self.content == "rels":
            return self.num_rels
        
    def getbatch_nodes_bert_id(self, raw_batch):
        nodes_str = []
        for i in range(len(raw_batch)):
            nodeid = raw_batch[i]           
            node_str = self.basedata.id2ent[nodeid]
            nodes_str.append(node_str)       
             
        nodes_code = self.tokenizer.batch_encode_plus(nodes_str, padding='longest', truncation=True, max_length=20)
        nodes_bert_id = torch.LongTensor(nodes_code["input_ids"])
        return nodes_bert_id
    
    def getbatch_rels_bert_id(self, raw_batch):
        rels_str = []
        for i in range(len(raw_batch)):
            relid = raw_batch[i]           
            rel_str = self.basedata.id2rel[relid]
            rels_str.append(rel_str)       
             
        rels_code = self.tokenizer.batch_encode_plus(rels_str, padding='longest', truncation=True, max_length=20)
        rels_bert_id = torch.LongTensor(rels_code["input_ids"])
        return rels_bert_id



def get_bert_init_embs(args, basedata, content):
    # device = args.device      #  GPU out of memory
    device = 'cpu'
    
    # model
    if args.models in ['bert']:
        if args.bert_model_dir != None:
            model = BertModel.from_pretrained(args.bert_model_dir).to(device)
        else:
            model = BertModel.from_pretrained(args.bert_model_name).to(device)
    else:
        raise NotImplementedError

    print("getting initialized embeddings about", content, ".....")
    dataset = NodesRels_dataset(args, basedata, content)
    if content == "nodes":
        emb_file = args.nodes_emb_file
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, 
                                            collate_fn=dataset.getbatch_nodes_bert_id)
    elif content == "rels":
        emb_file = args.rels_emb_file
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, 
                                            collate_fn=dataset.getbatch_rels_bert_id)
    else:
        raise Exception
    
    file_name_list = []
    with torch.no_grad():
        for i, batch_bert_id in enumerate(dataloader):
            batch_bert_id = batch_bert_id.to(device)  
            atten_masks = (batch_bert_id > 0)
            outputs = model(batch_bert_id, atten_masks)
            
            # 取CLS的表示
            # out = outputs.pooler_output # (batch_size, hidden_size) 
            
            # 取平均, 不包括CLS，SEP和PAD的部分
            atten_masks = (batch_bert_id > 0) & (batch_bert_id != dataset.CLS_ID) & (batch_bert_id != dataset.SEP_ID)
            last_hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
            hidden_size = last_hidden_state.shape[2]
            masks_hidden = atten_masks.unsqueeze(2).repeat(1,1,hidden_size)
            avg_num = torch.sum(atten_masks, 1)
            zeros = torch.zeros_like(last_hidden_state)
            last_hidden_state_new = torch.where(masks_hidden, last_hidden_state, zeros)
            out = torch.sum(last_hidden_state_new,1)
            out = torch.div(out, avg_num.unsqueeze(1))
            
            if i > 0 and i % 90 == 0:
                emb = emb.cpu().detach().numpy()
                file_name = emb_file[:-4]+str(i)+'.npy'
                file_name_list.append(file_name)
                np.save(file_name, emb)
                print("save part of embeddings in:",file_name)
                emb = out.cpu()
                continue
            if i == 0:
                emb = out.cpu()
            else:
                emb = torch.cat((emb,out.cpu()), 0)
    emb = emb.cpu().detach().numpy()
    file_name = emb_file[:-4]+str(i)+'.npy'
    file_name_list.append(file_name)
    np.save(file_name, emb)
    print("save part of embeddings in :",file_name)
    print('----------- total batch:', i)
    
    for i, file_name in enumerate(file_name_list):
        if i == 0:
            emb = np.load(file_name)
        else:
            embadd = np.load(file_name)
            emb = np.concatenate((emb, embadd), axis=0)
        os.remove(file_name)
        
    np.save(emb_file, emb)
    print("getting initialized embeddings about ", content, ":done.")

