#### Import all the supporting classes
import os
import random
import numpy as np
import torch.nn.functional as F
import torch
import json
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import AutoModel
from sklearn.metrics import accuracy_score

from bert_init_embs import get_bert_init_embs
from data import *
from evaluate import *
from parse_args import *
from get_canonical_rps import *
from models import *
from get_canonical_triples import *
from main_rank import testmodel

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


"""
rerank 阶段, 使用语言模型, 将测试数据  h,r,t 与 根据 h,r 检索到的训练三元组拼一起作为输入得到预测;
微调 该语言模型
"""

def testmodel_Bert_finetune(model, dataloader_test, args):
    """
    used to calculate accuracy of the fine-tuned model
    """
    with torch.no_grad():
        model.eval() 
        for i, data_batch_test in enumerate(dataloader_test):
            data_batch_dataNone = {k:v for k,v in data_batch_test.items() if v== None}
            data_batch_test = {k:v.to(args.device) for k,v in data_batch_test.items() if v!= None}
            data_batch_test.update(data_batch_dataNone)

            test_scores_batch = model(data_batch_test)                   # (bs, num_nodes)
            
            if i == 0:
                test_scores = test_scores_batch.cpu()
            else:
                test_scores = torch.cat((test_scores,test_scores_batch.cpu()), 0)
    return test_scores.data.numpy().tolist() # test_scores.data.cpu().numpy()


def testmodel_rerank_Bert(model, dataloader_test, args, integration_mth_2stage):
    """
    类似 testmodel_rerank_v4 
    """
    with torch.no_grad():
        model.eval() 
        for i, data_batch_test in enumerate(dataloader_test):
            data_batch_dataNone = {k:v for k,v in data_batch_test.items() if v== None}
            data_batch_test = {k:v.to(args.device) for k,v in data_batch_test.items() if v!= None}
            data_batch_test.update(data_batch_dataNone)
            
            test_scores_TopK = model(data_batch_test)                   # (bs, num_nodes)
            
            test_scores_TopK_raw = data_batch_test['test_scores_sorted'][:args.rerank_Top_K]  
                 
            test_scores_TopK_raw1 = torch.sigmoid(test_scores_TopK_raw)/torch.max(torch.sigmoid(test_scores_TopK_raw))
            test_scores_TopK1 = test_scores_TopK
            test_scores_TopK_final = test_scores_TopK_raw1 * args.omega + test_scores_TopK1 * (1-args.omega) 
            
            scores_TopK_sorted, sorts_TopK_indices = torch.sort(test_scores_TopK_final, dim=0, descending=True) 
            test_sorts_TopK = data_batch_test['test_sorts_indices'][:args.rerank_Top_K][sorts_TopK_indices]
            test_sorts_indices = torch.cat([test_sorts_TopK, data_batch_test['test_sorts_indices'][args.rerank_Top_K:]], dim=0).unsqueeze(0)

            if i == 0:
                # 重排之前的排序和分数
                test_sorts_raw_K = data_batch_test['test_sorts_indices'][:args.rerank_Top_K].cpu()
                test_scores_raw_K = test_scores_TopK_raw.cpu()
                test_scores_K = test_scores_TopK.cpu()
                # 处理后的得分
                test_scores_raw1_K = test_scores_TopK_raw1.cpu()
                test_scores1_K = test_scores_TopK1.cpu()
                test_scores_final_K = test_scores_TopK_final.cpu()
                # 重排后的排序和分数
                test_sorts_rerank = test_sorts_indices.cpu()        
                test_scores_final_K_rerank = scores_TopK_sorted.cpu()
            else:
                # 重排之前的排序和分数
                test_sorts_raw_K = torch.cat((test_sorts_raw_K, data_batch_test['test_sorts_indices'][:args.rerank_Top_K].cpu()), 0)
                test_scores_raw_K = torch.cat((test_scores_raw_K, test_scores_TopK_raw.cpu()), 0)
                test_scores_K = torch.cat((test_scores_K, test_scores_TopK.cpu()), 0)
                # 处理后的得分
                test_scores_raw1_K = torch.cat((test_scores_raw1_K, test_scores_TopK_raw1.cpu()), 0)
                test_scores1_K = torch.cat((test_scores1_K, test_scores_TopK1.cpu()), 0)
                test_scores_final_K = torch.cat((test_scores_final_K, test_scores_TopK_final.cpu()), 0)
                # 重排后的排序和分数
                test_sorts_rerank = torch.cat((test_sorts_rerank, test_sorts_indices.cpu()), 0)
                test_scores_final_K_rerank = torch.cat((test_scores_final_K_rerank, scores_TopK_sorted.cpu()), 0)
    details = [test_sorts_raw_K, test_scores_raw_K, test_scores_K, test_scores_raw1_K, test_scores1_K, test_scores_final_K, test_scores_final_K_rerank]
    details = [i.reshape(-1, args.rerank_Top_K).data.numpy() for i in details]
    return test_sorts_rerank.data.numpy(), details 


class BertModel_Rerank(nn.Module):
    def __init__(self, args):
        super(BertModel_Rerank, self).__init__()
        self.args = args
        self.rerank_bert_mth = args.rerank_bert_mth
        
        if args.bert_model_dir != None:
            self.bert_model = AutoModel.from_pretrained(args.bert_model_dir).to(args.device)
        else:
            self.bert_model = AutoModel.from_pretrained(args.bert_model_name).to(args.device)

        # self.dropout = torch.nn.Dropout(p=args.dropout)     # TODO
        self.dropout = nn.Dropout(0.2)
        if self.rerank_bert_mth in ['SUM', 'CLS'] :
            self.fc = torch.nn.Linear(self.args.bert_dim, 1)
            if self.rerank_bert_mth == 'SUM':
                self.lin_comb = nn.Parameter(torch.zeros(13, dtype=torch.float32, requires_grad=True))
                self.softmax = torch.nn.Softmax(dim=0)
        elif self.rerank_bert_mth in ['CLS_new'] :  # 是按照DistilBertForSequenceClassification 来写的
            self.pre_classifier = nn.Linear(args.bert_dim, args.bert_dim)
            self.classifier = nn.Linear(args.bert_dim, 1)
            

    
    def forward(self, data):
        x = self.bert_model(data["rerank_inputs"], output_hidden_states=True)   
        if self.rerank_bert_mth == 'CLS':
            comb_embedded = x.last_hidden_state[:,0]                                             # [1] 是pooler_output (cls token): (batch_size, hidden_size)
            x = self.fc(self.dropout(comb_embedded)).squeeze(-1)        # 经过fc 后 预测得分
            x = torch.sigmoid(x) 
        elif self.rerank_bert_mth == 'SUM':
            x = x.hidden_states                                                         # [2] 是hidden states: (batch_size, sequence_length, hidden_size
            scalars = self.softmax(self.lin_comb)
            embedded = torch.stack(
                [sc*x[idx][:,0,:] for (sc, idx) in zip(scalars, range(len(x)))], dim=1)
            comb_embedded = torch.sum(embedded, dim=1)                  # 每一层CLS的表示加权求和
            x = self.fc(self.dropout(comb_embedded)).squeeze(-1)        # 经过fc 后 预测得分
            x = torch.sigmoid(x)  
        elif self.rerank_bert_mth == 'CLS_new': # 是按照DistilBertForSequenceClassification 来写的
            hidden_state = x.last_hidden_state  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)
            x = self.classifier(pooled_output)  # (bs, num_labels)       # 经过fc 后 预测得分
            x = torch.sigmoid(x.squeeze(-1))       
        return x
        


def main_rerank_wo_kfs(args):
    ######################################## 使用模型一测试 所有数据 ########################################
    if args.saved_model_name_rank == '':
        assert False, "Please provide pre-trained checkpoint"
    else:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(args.model_path)
        if args.saved_model_path_rerank == '':       
            args.save_path = os.path.join(args.saved_model_name_rank, 'rerank_wo_kfs'+'_'+args.finetune_rerank_model_name+'_'+str(args.rerank_training_neg_K)+'_'+time.strftime("%Y-%m-%d_%H-%M-%S"))
            if args.save_path and not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
        else:
            args.save_path = args.saved_model_path_rerank
        saved_model_path_rerank = os.path.join(args.save_path, "rerank_wo_kfs_model.pth")
        
            
    set_logger(args)
    logging.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))      
    
    basedata = load_basedata(args)
    args.num_nodes = basedata.num_nodes
    args.num_rels = basedata.num_rels

    if torch.cuda.is_available():
        args.device = 'cuda:' + str(args.cuda)
    else:
        args.device = 'cpu'
    
    #####  bert init
    if args.bert_init:
        logging.info("utilizing bert initialized nodes(nps) embeddings.")
        logging.info("nodes_emb_file: %s " % args.nodes_emb_file)
        logging.info("utilizing bert initialized rels(rps) embeddings.")  
        logging.info("rels_emb_file: %s " % args.rels_emb_file)
        if not os.path.exists(args.nodes_emb_file):
            get_bert_init_embs(args, basedata, "nodes")
        if not os.path.exists(args.rels_emb_file):
            get_bert_init_embs(args, basedata, "rels")
    
    ##### canonical neighbor RPs
    if args.rp_neigh_mth == 'Local':  
        if not os.path.exists(args.data_files["rpneighs_filepath"]):    
            logging.info("getting canonical neighbor RPs...")
            get_canonical_rps(args, basedata, "new_sbert_ambv2", mth='ENT')   
        logging.info("reading canonical neighbor RPs file: " + args.data_files["rpneighs_filepath"])
    basedata.read_canonical_rps()
    args.num_rels = basedata.num_rels

    logging.info('Model: %s' % args.model_name)
    logging.info('Data Path: %s' % args.data_path)

    dataset_train = myDataset(args, basedata, "train")
    dataset_valid = myDataset(args, basedata, "valid")
    dataset_test = myDataset(args, basedata, "test")

    if args.rp_neigh_mth == 'Global':
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, 
                                        num_workers=args.num_workers,collate_fn=dataset_train.get_batch_Global_rps_train)  
        dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.num_workers,collate_fn=dataset_valid.get_batch_Global_rps_test) 
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.num_workers,collate_fn=dataset_test.get_batch_Global_rps_test)
    elif args.rp_neigh_mth == 'Local':
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, 
                                        num_workers=args.num_workers,collate_fn=dataset_train.get_batch_Local_rps_train)  
        dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.num_workers,collate_fn=dataset_valid.get_batch_Local_rps_test) 
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.num_workers,collate_fn=dataset_test.get_batch_Local_rps_test)    
    else:
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, 
                                        num_workers=args.num_workers,collate_fn=dataset_train.get_batch_train)  
        dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.num_workers,collate_fn=dataset_valid.get_batch_test) 
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, 
                                        num_workers=args.num_workers,collate_fn=dataset_test.get_batch_test)
        
    edges_np = torch.tensor(basedata.edges, dtype=torch.long).to(args.device)     
    node_id = torch.arange(0, basedata.num_nodes, dtype=torch.long).to(args.device)
    if args.model_name == "BertResNet_2Inp": 
        model_1 = BertResNet_2Inp(args, edges_np, node_id)
    elif args.model_name == "BertResNet_3Inp": 
        model_1 = BertResNet_3Inp(args, edges_np, node_id)
    elif args.model_name == "ConvE": 
        model_1 = ConvE(args, edges_np, node_id)
    else:
        raise "Model not implemented. Choose in [BertResNet_2Inp, BertResNet_3Inp, ConvE]"
    
    model_1 = model_1.to(args.device)

    # Restore model from checkpoint directory
    logging.info('Loading checkpoint %s...' % args.model_path)
    checkpoint = torch.load(args.model_path)
    model_1.eval()
    model_1.load_state_dict(checkpoint['state_dict'])

    if args.saved_model_path_rerank == '':       
        ######################################## 获得模型二的训练数据(不使用Retrieval data,直接使用hrt的句子) ########################################

        ##### train ##### 
        logging.info('Evaluating on Training Dataset...')
        scores_model1_train_set = testmodel(model_1, dataloader_train, args)

        ##### valid #####
        logging.info('Evaluating on Valid Dataset...')
        scores_model1_valid_set = testmodel(model_1, dataloader_valid, args)
        valid_perf, valid_ranks, valid_ranked_cands = evaluate_perf(basedata.valid_trips, scores_model1_valid_set, args, basedata, K=10)
        logging.info("valid:  MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (valid_perf["mrr"], valid_perf["mr"], valid_perf["hits@"][1], valid_perf["hits@"][3], valid_perf["hits@"][10], valid_perf["hits@"][30], valid_perf["hits@"][50]))

        # 直接在线处理就行, 不使用retrirval数据          
        dataset_rerank_train = DatasetFromScores_ReRank_Bert(args, basedata, scores_model1_train_set, dataset_train.train_pairs, 'train')
        dataset_rerank_valid = DatasetFromScores_ReRank_Bert(args, basedata, scores_model1_valid_set, basedata.valid_trips, 'valid')
        dataloader_rerank_train = DataLoader(dataset=dataset_rerank_train, batch_size=16, shuffle=True, 
                                num_workers=args.num_workers,collate_fn=dataset_rerank_train.get_batch_rerank_data_bert_train_valid)
        dataloader_rerank_valid = DataLoader(dataset=dataset_rerank_valid, batch_size=16, shuffle=False, 
                                num_workers=args.num_workers,collate_fn=dataset_rerank_valid.get_batch_rerank_data_bert_train_valid)
        logging.info("rerank model: number of train samples: %d"%(len(dataset_rerank_train.rerank_inputs)))
        logging.info("rerank model: number of valid samples: %d"%(len(dataset_rerank_valid.rerank_inputs)))
        
        #################### fine-tuning ####################                                           
        # 对比实验, 所以要和rerank模型微调同一个PLM  
        args.bert_model_dir = '/opt/data/private/PretrainedBert/distilbert-base-uncased'
        args.bert_model_name = 'distilbert-base-uncased'

        model_2 = BertModel_Rerank(args)
        model_2 = model_2.to(args.device)
        
        logging.info("Fine-tuning rerank model ...")
        num_epochs = 5                
        optimizer = torch.optim.Adam(model_2.parameters(), lr=args.lr_finetune)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max',factor = 0.5, patience = 2)
        criterion = torch.nn.BCELoss()
        
        best_ACC = 0
        best_epoch = 0
        count = 0
        for epoch in range(num_epochs):
            model_2.train()
            if count >= args.early_stop: break
            epoch_loss = 0                                        
            
            for i,data_batch_train in enumerate(dataloader_rerank_train):
                data_batch_dataNone = {k:v for k,v in data_batch_train.items() if v== None}
                data_batch_train = {k:v.to(args.device) for k,v in data_batch_train.items() if v!= None}
                data_batch_train.update(data_batch_dataNone)

                optimizer.zero_grad()
                train_scores = model_2(data_batch_train)
                
                loss = criterion(train_scores, data_batch_train["labels"])

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model_2.parameters(), args.grad_norm)
                optimizer.step()
                epoch_loss += (loss.data).cpu().numpy()
            logging.info("epoch {}/{} total epochs, epoch_loss: {}".format(epoch+1,args.n_epochs,epoch_loss/i))

            valid_scores = testmodel_Bert_finetune(model_2, dataloader_rerank_valid, args)
            valid_scores = [round(s) for s in valid_scores]
            valid_acc = accuracy_score(dataset_rerank_valid.labels, valid_scores)
            logging.info("valid acc:%6f" % (valid_acc))
            if valid_acc>best_ACC:
                count = 0
                best_ACC = valid_acc
                best_epoch = epoch + 1
                logging.info("save model.")
                torch.save({'state_dict': model_2.state_dict(), 'epoch': epoch}, saved_model_path_rerank)
            else:
                count+=1
            logging.info("Best Valid ACC: {}, Best Epoch: {}".format(best_ACC,best_epoch))
            scheduler.step(best_epoch)
    
        logging.info("Fine-tuning rerank model over.")
        logging.info("saved_model_path_rerank:", saved_model_path_rerank)

    
        #################### 使用模型二结合模型一进行rerank, 在 valid 上确定 omega 的值 ####################    
        
        logging.info("------- Validation Set Evaluation After Reranking -------")

        model_2 = BertModel_Rerank(args)
        model_2 = model_2.to(args.device)
        checkpoint = torch.load(saved_model_path_rerank)
        model_2.eval()
        model_2.load_state_dict(checkpoint['state_dict'])

        dataset_rerank_valid_new = DatasetFromScores_ReRank_Bert(args, basedata, scores_model1_valid_set, basedata.valid_trips, 'test')
        dataloader_rerank_valid_new = DataLoader(dataset=dataset_rerank_valid_new, batch_size=1, shuffle=False, 
                                num_workers=args.num_workers,collate_fn=dataset_rerank_valid_new.get_batch_rerank_data_bert_test)
        bestMRR_im_omega = {'integration_mth_2stage':-1, 'omega':-1, 'mrr':-1, 'all':''}
        for integration_mth_2stage in [1]:   # 方法 3 只需要跑一次,不需要omega
            logging.info("--------- integration_mth_2stage:%d ---------"%integration_mth_2stage)
            best_omega = {'omega':-1, 'mrr':0, 'all':''}
            for i in range(1,10):
                args.omega = i * 0.1
                logging.info(" ")
                logging.info("omega:%3f" % (args.omega))

                valid_sorts, valid_details = testmodel_rerank_Bert(model_2, dataloader_rerank_valid_new, args, integration_mth_2stage)  
                valid_perf, valid_ranks, valid_ranked_cands = evaluate_perf_sort(basedata.valid_trips, valid_sorts, args, basedata, K=args.rerank_Top_K)
                valid_info = "valid:  MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (valid_perf["mrr"], valid_perf["mr"], valid_perf["hits@"][1], valid_perf["hits@"][3], valid_perf["hits@"][10], valid_perf["hits@"][30], valid_perf["hits@"][50])
                logging.info(valid_info)
                if valid_perf["mrr"] > best_omega["mrr"]:
                    best_omega["omega"], best_omega["mrr"], best_omega["all"] = args.omega, valid_perf["mrr"], valid_info
            logging.info("--------- best: omega:%3f, MRR:%6f ---------"%(best_omega["omega"], best_omega["mrr"]))
            logging.info("--------- best: %s ---------"%(best_omega["all"]))
            if best_omega["mrr"] > bestMRR_im_omega["mrr"]:
                bestMRR_im_omega["integration_mth_2stage"], bestMRR_im_omega["omega"], bestMRR_im_omega["mrr"], bestMRR_im_omega["all"] = integration_mth_2stage, best_omega["omega"], best_omega["mrr"], best_omega["all"]
        logging.info("-------- best: integration_mth_2stage:%d, omega:%3f, MRR:%6f"%(bestMRR_im_omega["integration_mth_2stage"], bestMRR_im_omega["omega"], bestMRR_im_omega["mrr"]))
        logging.info("-------- best: %s"%(bestMRR_im_omega["all"]))

        integration_mth_2stage = bestMRR_im_omega["integration_mth_2stage"]
        args.omega = bestMRR_im_omega["omega"]        
    else:
        #################### 直接进行测试 ####################
        model_2 = BertModel_Rerank(args)
        model_2 = model_2.to(args.device)
        checkpoint = torch.load(saved_model_path_rerank)
        model_2.eval()
        model_2.load_state_dict(checkpoint['state_dict'])
        
    #################### 测试 ####################
    ##### test data #####
    logging.info('Evaluating on Test Dataset...')
    scores_model1_test_set = testmodel(model_1, dataloader_test, args, logtime=True)
    test_perf, test_ranks, test_ranked_cands = evaluate_perf(basedata.test_trips, scores_model1_test_set, args, basedata, K=10)
    logging.info("test:  MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))
    # save_preds(basedata.test_trips, test_ranked_cands, basedata, None, args.preds_path_txt % "test")

    dataset_rerank_test = DatasetFromScores_ReRank_Bert(args, basedata, scores_model1_test_set, basedata.test_trips, 'test')
    dataloader_rerank_test = DataLoader(dataset=dataset_rerank_test, batch_size=1, shuffle=False, 
                                num_workers=args.num_workers,collate_fn=dataset_rerank_test.get_batch_rerank_data_bert_test)
    ##### testing #####
    logging.info("integration_mth_2stage:%d" % (integration_mth_2stage))
    logging.info("omega:%3f" % (args.omega))
    test_sorts, test_details = testmodel_rerank_Bert(model_2, dataloader_rerank_test, args, integration_mth_2stage)  
    test_perf, test_ranks, test_ranked_cands = evaluate_perf_sort(basedata.test_trips, test_sorts, args, basedata, K=args.rerank_Top_K)
    # save_ranks(basedata.test_trips, test_ranks, args.ranks_path % "test")
    # save_preds(basedata.test_trips, test_ranked_cands, basedata, args.preds_path_npy % "test", args.preds_path_txt % "test")
    # save_preds_rerank_details(basedata.test_trips, test_ranked_cands, basedata, test_details, args.preds_path_txt % ("test_"+str(integration_mth_2stage)+"_"+str(args.omega)))
    logging.info("after rerank, test:  MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))
    
    


 
if __name__ == '__main__':
    parser = getParser()
    args = parser.parse_args()
    args = set_params(args)

    seed = args.seed 
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main_rerank_wo_kfs(args)
