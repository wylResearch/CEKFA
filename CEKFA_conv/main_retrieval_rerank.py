#### Import all the supporting classes
from encodings import search_function
from datetime import datetime 
import os
import math
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from rank_bm25 import BM25Okapi
from tqdm.autonotebook import tqdm

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample


from bert_init_embs import get_bert_init_embs
from data import *
from evaluate import *
from parse_args import *
from get_canonical_rps import *
from models import *
from get_canonical_triples import *
from main_rank import testmodel, main_rank

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


"""
rerank 阶段, 使用语言模型, 将测试数据  h,r,t 与 根据 h,r 检索到的训练三元组拼一起作为输入得到预测;
微调 该语言模型
"""

def testmodel_rerank_retrieval(model, dataloader_test, args, integration_mth_2stage, logtime=False):
    """
    类似 testmodel_rerank_v4 
    """
    if logtime:
        logging.info("start test...")
    total_infer_time_ba = 0
    total_infer_time_ca = 0
    total_infer_time_da = 0
    for i, data_batch_test in enumerate(dataloader_test):
        a=datetime.now() 
        test_scores_TopK = torch.tensor(model.predict(data_batch_test['rerank_input']))      # 0-1之间的概率值,不需要sigmoid     
        b=datetime.now()   
        
        test_scores_TopK_raw = data_batch_test['test_scores_sorted'][:args.rerank_Top_K]  
        
        test_scores_TopK_raw1 = F.sigmoid(test_scores_TopK_raw)/torch.max(F.sigmoid(test_scores_TopK_raw))
        
        test_scores_TopK1 = test_scores_TopK
        test_scores_TopK_final = test_scores_TopK_raw1 * args.omega + test_scores_TopK1 * (1-args.omega) 

        scores_TopK_sorted, sorts_TopK_indices = torch.sort(test_scores_TopK_final, dim=0, descending=True) 
        test_sorts_TopK = data_batch_test['test_sorts_indices'][:args.rerank_Top_K][sorts_TopK_indices]
        test_sorts_indices = torch.cat([test_sorts_TopK, data_batch_test['test_sorts_indices'][args.rerank_Top_K:]], dim=0).unsqueeze(0)
        c=datetime.now()   
        total_infer_time_ba += (b-a).total_seconds()
        total_infer_time_ca += (c-a).total_seconds()

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
        # d=datetime.now()   
        # total_infer_time_da += (d-a).total_seconds()
    details = [test_sorts_raw_K, test_scores_raw_K, test_scores_K, test_scores_raw1_K, test_scores1_K, test_scores_final_K, test_scores_final_K_rerank]
    details = [i.reshape(-1, args.rerank_Top_K).data.numpy() for i in details]
    if logtime:
        logging.info("end test...")
        num_samples = details[0].shape[0]
        logging.info("num_samples:%s"%num_samples)
        logging.info("total_infer_time_ba:%s"%total_infer_time_ba)
        logging.info("total_infer_time_ba/num_samples:%f"%(total_infer_time_ba/num_samples))
        logging.info("num_samples/total_infer_time_ba:%f"%(num_samples/total_infer_time_ba))
        logging.info("total_infer_time_ca:%s"%total_infer_time_ca)
        logging.info("total_infer_time_ca/num_samples:%f"%(total_infer_time_ca/num_samples))
        logging.info("num_samples/total_infer_time_ca:%f"%(num_samples/total_infer_time_ca))
        # logging.info("total_infer_time_da:%s"%total_infer_time_da)
        # logging.info("total_infer_time_da/num_samples:%f"%(total_infer_time_da/num_samples))
        # logging.info("num_samples/total_infer_time_da:%f"%(num_samples/total_infer_time_da))
    return test_sorts_rerank.data.numpy(), details 


def get_input_examples_hr(args, basedata, scores, trips, retrieval_datas, mode, sep_id_rp):
    """
    微调CrossEncoder,训练和验证数据需要用InputExample生成. 
    该方法中 检索相关文档的 key是 hr, value 是 相关的 hr 对应的 hrt
    """
    samples = []

    retrieval_data_dict = {}
    for item in retrieval_datas:
        retrieval_data_dict[(item[0],item[1])] = item[2]
    
    for i in range(len(trips)):   
        score, trip = scores[i], trips[i]
        if mode == 'train':
            h, r = trip   
            true_tails = basedata.label_graph[(h,r)]
        else:
            h, r, t = trip    
            true_tails = basedata.label_graph[(h,r)] if (h,r) in basedata.label_graph else set()
            true_tails = true_tails.union(set([t]))
        
        retrieval_data = retrieval_data_dict[(h,r)]
        
        scores_sorted, sorts_indices = torch.sort(torch.tensor(score), dim=0, descending=True)
        candi_TopK = sorts_indices[:(args.rerank_training_neg_K+len(true_tails))].tolist()
        candi_TopK_neg = []
        for candi in candi_TopK:
            if len(candi_TopK_neg) == args.rerank_training_neg_K:
                break
            if candi not in true_tails:
                candi_TopK_neg.append(candi)
 
        if r < sep_id_rp:
            for t in true_tails:
                rerank_input = [basedata.id2ent[h] + ' ' + basedata.id2rel[r] + ' '+ basedata.id2ent[t], retrieval_data]
                samples.append(InputExample(texts=rerank_input, label=1.0))
            for t in candi_TopK_neg:
                rerank_input = [basedata.id2ent[h] + ' ' + basedata.id2rel[r] + ' '+ basedata.id2ent[t], retrieval_data]
                samples.append(InputExample(texts=rerank_input, label=0.0))
        else:
            r_inv = basedata.inverse2rel_map[r]  
            for t in true_tails:
                rerank_input = [basedata.id2ent[t] + ' ' + basedata.id2rel[r_inv] + ' '+ basedata.id2ent[h], retrieval_data]
                samples.append(InputExample(texts=rerank_input, label=1.0))      
            for t in candi_TopK_neg:
                rerank_input = [basedata.id2ent[t] + ' ' + basedata.id2rel[r_inv] + ' '+ basedata.id2ent[h], retrieval_data]
                samples.append(InputExample(texts=rerank_input, label=0.0))
    return samples


def main_rerank(args):
    ######################################## 使用模型一测试 所有数据 ########################################
    if args.saved_model_name_rank == '':
        assert False, "Please provide pre-trained checkpoint"
    else:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(args.model_path)
        if args.saved_model_path_rerank == '':       
            saved_model_path_rerank = os.path.join(args.saved_model_name_rank, 'new_rerank'+'_'+args.finetune_rerank_model_name+'_'+args.retrieval_dirname+'_'+args.retrieval_version+'_'+str(args.retrieval_Top_K)+'_'+str(args.rerank_training_neg_K)+'_'+time.strftime("%Y-%m-%d_%H-%M-%S"))
            args.save_path = saved_model_path_rerank
            if args.save_path and not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
        else:
            args.save_path = args.saved_model_path_rerank
            saved_model_path_rerank = args.saved_model_path_rerank
            

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

    ########################### 使用Retrieval的数据进行rerank ###########################
    sep_id_rp = len(basedata.rel2id) / 2 if args.reverse else len(basedata.rel2id)         
    retrieval_Top_K = args.retrieval_Top_K
    retrieval_datadir = os.path.join(args.retrieval_basedir, args.retrieval_dirname)  
    files_path = [os.path.join(retrieval_datadir, "retrieval_" + args.retrieval_version + "_" + split + '_Top' + str(retrieval_Top_K) +'.txt') for split in ['train', 'valid', 'test']]
    files_exist = [os.path.exists(file) for file in files_path]
    if sum(files_exist) != 3:
        logging.info("search and write similar training data for each query.")
        if args.retrieval_dirname == 'bm25':
            get_canonical_triples_BM25(args, basedata)
        else:
            get_canonical_triples_sbert(args, basedata)
    logging.info("read search results from: %s" % (retrieval_datadir))
    retrieval_data_list = read_canonical_triples(retrieval_datadir, args.retrieval_version, retrieval_Top_K)

    if args.saved_model_path_rerank == '':       
        #################### 获得模型二的训练数据 (Retrieval data) 用于微调 ####################

        ##### train ##### 
        logging.info('Evaluating on Training Dataset...')
        scores_model1_train_set = testmodel(model_1, dataloader_train, args)

        ##### valid #####
        logging.info('Evaluating on Valid Dataset...')
        scores_model1_valid_set = testmodel(model_1, dataloader_valid, args)
        valid_perf, valid_ranks, valid_ranked_cands = evaluate_perf(basedata.valid_trips, scores_model1_valid_set, args, basedata, K=10)
        logging.info("valid:  MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (valid_perf["mrr"], valid_perf["mr"], valid_perf["hits@"][1], valid_perf["hits@"][3], valid_perf["hits@"][10], valid_perf["hits@"][30], valid_perf["hits@"][50]))

        logging.info("construct training examples ...")
        train_samples = get_input_examples_hr(args, basedata, scores_model1_train_set, dataset_train.train_pairs, retrieval_data_list[0], 'train', sep_id_rp)
        logging.info("construct validation examples ...")
        valid_samples = get_input_examples_hr(args, basedata, scores_model1_valid_set, basedata.valid_trips, retrieval_data_list[1], 'valid', sep_id_rp)
        logging.info("rerank model: number of train samples: %d"%(len(train_samples)))
        logging.info("rerank model: number of valid samples: %d"%(len(valid_samples)))

        #################### fine-tuning ####################                                           
        if args.finetune_rerank_model_dir != '':
            args.finetune_rerank_model_name = args.finetune_rerank_model_dir.split("/")[-1]
            model_2 = CrossEncoder(args.finetune_rerank_model_dir, num_labels=1)      # 封装的 AutoModelForSequenceClassification       以前是 'distilroberta-base' 现在是 以前是 'distilbert-base-uncased'
        else:
            model_2 = CrossEncoder(args.finetune_rerank_model_name, num_labels=1)      # 封装的 AutoModelForSequenceClassification       以前是 'distilroberta-base' 现在是 以前是 'distilbert-base-uncased'
    
        logging.info("Fine-tuning rerank model ...")
        num_epochs = 5
        # train_samples, valid_samples = valid_samples[:10], valid_samples[:10]     

        train_batch_size = 16
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        evaluator = CEBinaryAccuracyEvaluator.from_input_examples(valid_samples, name=args.dataset + ' valid') # CEBinaryClassificationEvaluator
        
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

        # Train the model
        model_2.fit(train_dataloader=train_dataloader,
                evaluator=evaluator,
                epochs=num_epochs,
                evaluation_steps=10000,
                warmup_steps=warmup_steps,
                output_path=saved_model_path_rerank)
        
        logging.info("Fine-tuning rerank model over.")
        logging.info("saved_model_path_rerank:", saved_model_path_rerank)

        #################### 使用模型二结合模型一进行rerank, 在 valid 上确定 omega 的值 ####################    
        dataset_rerank_valid = DatasetFromScores_Retrieval_ReRank(args, basedata, scores_model1_valid_set, basedata.valid_trips, data1=retrieval_data_list[1])
        dataloader_rerank_valid = DataLoader(dataset=dataset_rerank_valid, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers,collate_fn=dataset_rerank_valid.get_batch_retrival_data)
        
        logging.info("Test-Reranking ...")    
        bestMRR_im_omega = {'integration_mth_2stage':-1, 'omega':-1, 'mrr':-1, 'all':''}
        for integration_mth_2stage in [1]:   # 方法 3 只需要跑一次,不需要omega
            logging.info("--------- integration_mth_2stage:%d ---------"%integration_mth_2stage)
            best_omega = {'omega':-1, 'mrr':0, 'all':''}
            for i in range(1,10):
                args.omega = i * 0.1
                logging.info(" ")
                logging.info("omega:%3f" % (args.omega))

                valid_sorts, valid_details = testmodel_rerank_retrieval(model_2, dataloader_rerank_valid, args, integration_mth_2stage)  
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
        model_2 = CrossEncoder(args.saved_model_path_rerank)
        integration_mth_2stage = 1
        # Note: set value for args.omega

    #################### 测试 ####################
    ##### test data #####
    logging.info('Evaluating on Test Dataset...')
    scores_model1_test_set = testmodel(model_1, dataloader_test, args, logtime=True)
    test_perf, test_ranks, test_ranked_cands = evaluate_perf(basedata.test_trips, scores_model1_test_set, args, basedata, K=10)
    logging.info("test:  MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))

    dataset_rerank_test = DatasetFromScores_Retrieval_ReRank(args, basedata, scores_model1_test_set, basedata.test_trips, data1=retrieval_data_list[2])
    dataloader_rerank_test = DataLoader(dataset=dataset_rerank_test, batch_size=1, shuffle=False, 
                                num_workers=args.num_workers,collate_fn=dataset_rerank_test.get_batch_retrival_data)
        
    ##### testing #####
    logging.info("integration_mth_2stage:%d" % (integration_mth_2stage))
    logging.info("omega:%3f" % (args.omega))
    test_sorts, test_details = testmodel_rerank_retrieval(model_2, dataloader_rerank_test, args, integration_mth_2stage, logtime=True)  
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

    main_rerank(args)
