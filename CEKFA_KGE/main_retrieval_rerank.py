#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import logging
import os
import random
import time
import math
import numpy as np
import torch
from datetime import datetime 

from torch.utils.data import DataLoader
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator, CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample

from models import KGEModel
from bert_init_embs import get_bert_init_embs
from helper import *
from data import *
from parse_args import *
from get_canonical_rps import *
from get_canonical_triples import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def testmodel_rerank_retrieval(model, dataloader_test, args, integration_mth_2stage, logger, logtime=False):
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
        
        test_scores_TopK_raw1 = normalization(test_scores_TopK_raw)

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


def get_input_examples_hr_KGE(args, basedata, scores, trips, retrieval_datas, mode):
    """
    微调CrossEncoder,训练和验证数据需要用InputExample生成. 
    该方法中 检索相关文档的 key是 hr, value 是 相关的 hr 对应的 hrt
    """
    head_scores, tail_scores = scores
    samples = []
    
    ####   tail-batch   # h,r,(t)
    for i in range(len(trips)):   
        score, trip = tail_scores[i], trips[i]
        if mode == 'train':
            h, r, t = trip   
            true_tails = basedata.label_graph[(h,r)]
        else:
            h, r, t = trip    
            true_tails = basedata.label_graph[(h,r)] if (h,r) in basedata.label_graph else set()
            true_tails = true_tails.union(set([t]))
        
        retrieval_data = retrieval_datas[2*i][2]
        
        scores_sorted, sorts_indices = torch.sort(torch.tensor(score), dim=0, descending=True)      # 用哪个排序
        candi_TopK = sorts_indices[:(args.rerank_training_neg_K+len(true_tails))].tolist()
        candi_TopK_neg = []
        for candi in candi_TopK:
            if len(candi_TopK_neg) == args.rerank_training_neg_K:
                break
            if candi not in true_tails:
                candi_TopK_neg.append(candi)

        for t_ in true_tails:
            rerank_input = [basedata.id2ent[h] + ' ' + basedata.id2rel[r] + ' '+ basedata.id2ent[t_], retrieval_data]
            samples.append(InputExample(texts=rerank_input, label=1.0))
        for t_ in candi_TopK_neg:
            rerank_input = [basedata.id2ent[h] + ' ' + basedata.id2rel[r] + ' '+ basedata.id2ent[t_], retrieval_data]
            samples.append(InputExample(texts=rerank_input, label=0.0))
    
    #### head-batch   # (h),r,t
    for i in range(len(trips)):   
        score, trip = head_scores[i], trips[i]
        if mode == 'train':
            h, r, t = trip   
            true_heads = basedata.label_graph_tr2h[(t,r)]
        else:
            h, r, t = trip    
            true_heads = basedata.label_graph_tr2h[(t,r)] if (t,r) in basedata.label_graph_tr2h else set()
            true_heads = true_heads.union(set([h]))
        
        retrieval_data = retrieval_datas[2*i+1][2]
        
        scores_sorted, sorts_indices = torch.sort(torch.tensor(score), dim=0, descending=True)      # 用哪个排序
        candi_TopK = sorts_indices[:(args.rerank_training_neg_K+len(true_heads))].tolist()
        candi_TopK_neg = []
        for candi in candi_TopK:
            if len(candi_TopK_neg) == args.rerank_training_neg_K:
                break
            if candi not in true_heads:
                candi_TopK_neg.append(candi)
 
        for h_ in true_heads:
            rerank_input = [basedata.id2ent[h_] + ' ' + basedata.id2rel[r] + ' '+ basedata.id2ent[t], retrieval_data]
            samples.append(InputExample(texts=rerank_input, label=1.0))      
        for h_ in candi_TopK_neg:
            rerank_input = [basedata.id2ent[h_] + ' ' + basedata.id2rel[r] + ' '+ basedata.id2ent[t], retrieval_data]
            samples.append(InputExample(texts=rerank_input, label=0.0))
    return samples


def normalization(scores):
    score_max = scores[0]
    score_min = scores[-1]
    score_range = score_max.sub(score_min)
    scores_ =  (scores - score_min)/score_range
    return scores_


def main_rerank(args):
    ######################################## 使用模型一测试 所有数据 ########################################
    if args.init_checkpoint is None:
        assert False, "Please provide pre-trained checkpoint"
    else:
        if args.saved_model_path_rerank == '':       
            saved_model_path_rerank = os.path.join(args.init_checkpoint, 'new_rerank'+'_'+args.finetune_rerank_model_name+'_'+args.retrieval_dirname+'_'+args.retrieval_version+'_'+str(args.retrieval_Top_K)+'_'+str(args.rerank_training_neg_K)+'_'+time.strftime("%Y-%m-%d_%H-%M-%S"))
            args.save_path = saved_model_path_rerank
            if args.save_path and not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
        else:
            args.save_path = args.saved_model_path_rerank
            args.do_train = False

    set_logger(args)

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
    args.rps2neighs, args.inverse2rel_map = basedata.rps2neighs, basedata.inverse2rel_map
    args.num_rels = basedata.num_rels
    num_nodes = args.num_nodes
    num_rels = args.num_rels
    
    logging.info('Model: %s' % args.model_name)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % num_nodes)
    logging.info('#relation: %d' % num_rels)
    
    train_trips = [tuple(item) for item in basedata.train_trips.tolist()]     
    logging.info('#train: %d' % len(train_trips))
    valid_trips = [tuple(item) for item in basedata.valid_trips.tolist()]         
    logging.info('#valid: %d' % len(valid_trips))
    test_trips = [tuple(item) for item in basedata.test_trips.tolist()]           
    logging.info('#test: %d' % len(test_trips))
    
    # All true triples
    all_true_triples = train_trips + valid_trips + test_trips
    
    edges_np = torch.tensor(basedata.edges, dtype=torch.long).to(args.device)     
    node_id = torch.arange(0, basedata.num_nodes, dtype=torch.long).to(args.device)
    kge_model = KGEModel(args, edges_np, node_id)
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = kge_model.to(args.device)
    
    # Restore model from checkpoint directory
    logging.info('Loading checkpoint %s...' % args.init_checkpoint)
    checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
    init_step = checkpoint['step']
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    
    step = init_step

    ########################### 使用Retrieval的数据进行rerank ###########################
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
        metrics, scores_model1_train_set = kge_model.test_step(kge_model, train_trips, all_true_triples, args)
        log_metrics('Test training data', step, metrics)
        train_perf, train_ranks, train_ranked_cands = evaluate_perf_kge(np.array(train_trips), scores_model1_train_set, args, basedata, K=10)
        logging.info("Train #### perf: %s"%(json.dumps(train_perf)))
        logging.info("Train #### MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (train_perf["mrr"], train_perf["mr"], train_perf["hits@"][1], train_perf["hits@"][3], train_perf["hits@"][10], train_perf["hits@"][30], train_perf["hits@"][50]))

        ##### valid #####
        logging.info('Evaluating on Valid Dataset...')
        metrics, scores_model1_valid_set = kge_model.test_step(kge_model, valid_trips, all_true_triples, args)
        log_metrics('Test valid data', step, metrics)
        valid_trips_head = np.array([[trip[2], trip[1], trip[0]] for trip in basedata.valid_trips])   # h,r,t, -> t,r,h
        valid_trips_all = np.concatenate((basedata.valid_trips, valid_trips_head),axis=1).reshape(-1, 3)
        valid_scores_head, valid_scores_tail = scores_model1_valid_set
        valid_scores = np.concatenate((valid_scores_tail,valid_scores_head),axis=1).reshape(-1, valid_scores_head.shape[1])
        valid_perf, valid_ranks, valid_ranked_cands = evaluate_perf_kge_v2(valid_trips_all, valid_scores, args, basedata, K=10)
        logging.info("Valid #### perf: %s"%(json.dumps(valid_perf)))
        logging.info("Valid #### MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (valid_perf["mrr"], valid_perf["mr"], valid_perf["hits@"][1], valid_perf["hits@"][3], valid_perf["hits@"][10], valid_perf["hits@"][30], valid_perf["hits@"][50]))

        logging.info("construct training examples ...")
        train_samples = get_input_examples_hr_KGE(args, basedata, scores_model1_train_set, train_trips, retrieval_data_list[0], 'train')
        logging.info("construct validation examples ...")
        valid_samples = get_input_examples_hr_KGE(args, basedata, scores_model1_valid_set, valid_trips, retrieval_data_list[1], 'valid')
        logging.info("rerank model: number of train samples: %d"%(len(train_samples)))
        logging.info("rerank model: number of valid samples: %d"%(len(valid_samples)))
        

        #################### fine-tuning ####################
        if args.finetune_rerank_model_dir != '':    # load pre-trained model from local dir
            args.finetune_rerank_model_name = args.finetune_rerank_model_dir.split("/")[-1]
            model_2 = CrossEncoder(args.finetune_rerank_model_dir, num_labels=1)      # 封装的 AutoModelForSequenceClassification       以前是 'distilroberta-base' 现在是 以前是 'distilbert-base-uncased'
        else:
            model_2 = CrossEncoder(args.finetune_rerank_model_name, num_labels=1)      # 封装的 AutoModelForSequenceClassification       以前是 'distilroberta-base' 现在是 以前是 'distilbert-base-uncased'
        
        logging.info("Fine-tuning rerank model ...")
        num_epochs = 5

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
        dataset_rerank_valid = DatasetFromScores_Retrieval_ReRank_KGE(args, basedata, valid_scores, valid_trips_all, data1=retrieval_data_list[1])
        dataloader_rerank_valid = DataLoader(dataset=dataset_rerank_valid, batch_size=1, shuffle=False, 
                                    num_workers=1,collate_fn=dataset_rerank_valid.get_batch_retrival_data)

        logging.info("Test-Reranking ...")    
        bestMRR_im_omega = {'integration_mth_2stage':-1, 'omega':-1, 'mrr':-1, 'all':''}
        for integration_mth_2stage in [1]:   # 方法 3 只需要跑一次,不需要omega
            logging.info("--------- integration_mth_2stage:%d ---------"%integration_mth_2stage)
            best_omega = {'omega':-1, 'mrr':0, 'all':''}
            for i in range(1,11):
                args.omega = i * 0.1
                logging.info(" ")
                logging.info("omega:%3f" % (args.omega))

                valid_sorts, valid_details = testmodel_rerank_retrieval(model_2, dataloader_rerank_valid, args, integration_mth_2stage, logging)  
                valid_perf, valid_ranks, valid_ranked_cands = evaluate_perf_sort_KGE(valid_trips_all, valid_sorts, args, basedata, K=args.rerank_Top_K)
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
    metrics, scores_model1_test_set = kge_model.test_step(kge_model, test_trips, all_true_triples, args)
    log_metrics('Test', step, metrics)
    test_trips_head = np.array([[trip[2], trip[1], trip[0]] for trip in basedata.test_trips])   # h,r,t, -> t,r,h
    test_trips_all = np.concatenate((basedata.test_trips, test_trips_head),axis=1).reshape(-1, 3)
    test_scores_head, test_scores_tail = scores_model1_test_set
    test_scores = np.concatenate((test_scores_tail,test_scores_head),axis=1).reshape(-1, test_scores_head.shape[1])
    test_perf, test_ranks, test_ranked_cands = evaluate_perf_kge_v2(test_trips_all, test_scores, args, basedata, K=10)
    logging.info("Test #### perf: %s"%(json.dumps(test_perf)))
    logging.info("Test #### MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))
    
    dataset_rerank_test = DatasetFromScores_Retrieval_ReRank_KGE(args, basedata, test_scores, test_trips_all, data1=retrieval_data_list[2])
    dataloader_rerank_test = DataLoader(dataset=dataset_rerank_test, batch_size=1, shuffle=False,  
                                num_workers=1,collate_fn=dataset_rerank_test.get_batch_retrival_data)

    ##### testing #####
    logging.info("integration_mth_2stage:%d" % (integration_mth_2stage))
    logging.info("omega:%3f" % (args.omega))
    test_sorts, test_details = testmodel_rerank_retrieval(model_2, dataloader_rerank_test, args, integration_mth_2stage, logging)  
    test_perf, test_ranks, test_ranked_cands = evaluate_perf_sort_KGE(test_trips_all, test_sorts, args, basedata, K=args.rerank_Top_K)
    # save_ranks(basedata.test_trips, test_ranks, args.ranks_path % "test")
    # save_preds(basedata.test_trips, test_ranked_cands, basedata, args.preds_path_npy % "test", args.preds_path_txt % "test")
    # save_preds_rerank_details(basedata.test_trips, test_ranked_cands, basedata, test_details, args.preds_path_txt % ("test_"+str(integration_mth_2stage)+"_"+str(args.omega)))
    logging.info("after rerank, test:  MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))
    


if __name__ == '__main__':
    args = parse_args()
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