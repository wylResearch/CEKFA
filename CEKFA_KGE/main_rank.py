#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import KGEModel

from bert_init_embs import get_bert_init_embs
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from helper import *
from data import load_basedata
from parse_args import *
from get_canonical_rps import *

   
        
def main_rank(args):
    # Write logs to checkpoint and console
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
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_trips, num_nodes, num_rels, args.negative_sample_size, 'head-batch', args.rps2neighs, args.inverse2rel_map), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_trips, num_nodes, num_rels, args.negative_sample_size, 'tail-batch', args.rps2neighs, args.inverse2rel_map), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    early_stop = 3
    if args.do_train:
        best_mrr = 0
        patience = 0
        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            # if step % args.save_checkpoint_steps == 0:
            #     save_variable_list = {
            #         'step': step, 
            #         'current_learning_rate': current_learning_rate,
            #         'warm_up_steps': warm_up_steps
            #     }
            #     save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics, valid_scores = kge_model.test_step(kge_model, valid_trips, all_true_triples, args)
                log_metrics('Valid', step, metrics)
                valid_perf, valid_ranks, valid_ranked_cands = evaluate_perf_kge(np.array(valid_trips), valid_scores, args, basedata, K=10)
                logging.info("Valid #### perf: %s"%(json.dumps(valid_perf)))
                logging.info("Valid #### MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (valid_perf["mrr"], valid_perf["mr"], valid_perf["hits@"][1], valid_perf["hits@"][3], valid_perf["hits@"][10], valid_perf["hits@"][30], valid_perf["hits@"][50]))

                if valid_perf["mrr"] > best_mrr:
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args)
                    best_mrr = valid_perf["mrr"]
                    patience = 0
                else:
                    patience += 1

            if patience == early_stop:
                break

        if not args.do_valid:
            save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
            save_model(kge_model, optimizer, save_variable_list, args)
        
        logging.info('Training over.')
        logging.info('Loading best model...')
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])

        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics, valid_scores = kge_model.test_step(kge_model, valid_trips, all_true_triples, args)
        log_metrics('Valid', step, metrics)
        valid_perf, valid_ranks, valid_ranked_cands = evaluate_perf_kge(np.array(valid_trips), valid_scores, args, basedata, K=10)
        logging.info("Valid #### perf: %s"%(json.dumps(valid_perf)))
        logging.info("Valid #### MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (valid_perf["mrr"], valid_perf["mr"], valid_perf["hits@"][1], valid_perf["hits@"][3], valid_perf["hits@"][10], valid_perf["hits@"][30], valid_perf["hits@"][50]))

    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics, test_scores = kge_model.test_step(kge_model, test_trips, all_true_triples, args)
        log_metrics('Test', step, metrics)
        test_perf, test_ranks, test_ranked_cands = evaluate_perf_kge(np.array(test_trips), test_scores, args, basedata, K=10)
        logging.info("Test #### perf: %s"%(json.dumps(test_perf)))
        logging.info("Test #### MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))

        file_path = 'results_' + args.dataset +'.txt'
        with open(file_path, mode='a', encoding='utf-8') as file_obj:
            file_obj.write("%s\tMRR:%6f\tMR:%6f\thits@1:%6f\thits@3:%6f\thits@10:%6f\thits@30:%6f\thits@50:%6f\n" % (args.save_dir_name, test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))

    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics, train_scores = kge_model.test_step(kge_model, train_trips, all_true_triples, args)
        log_metrics('Test', step, metrics)
        train_perf, train_ranks, train_ranked_cands = evaluate_perf_kge(np.array(train_trips), train_scores, args, basedata, K=10)
        logging.info("Train #### perf: %s"%(json.dumps(train_perf)))
        logging.info("Train #### MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (train_perf["mrr"], train_perf["mr"], train_perf["hits@"][1], train_perf["hits@"][3], train_perf["hits@"][10], train_perf["hits@"][30], train_perf["hits@"][50]))

        
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


    main_rank(args)
