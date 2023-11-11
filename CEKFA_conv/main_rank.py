#### Import all the supporting classes
import os
import sys
import time
import math
import uuid
import getpass
import random
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime 


from bert_init_embs import get_bert_init_embs
from data import *
from evaluate import *
from parse_args import *
from get_canonical_rps import *
from models import *


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def testmodel(model, dataloader_test, args, logtime=False):
    if logtime:
        logging.info("start test...")
    total_infer_time_ba = 0
    total_infer_time_ca = 0
    total_infer_time_da = 0
    with torch.no_grad():
        model.eval() 
        for i, data_batch_test in enumerate(dataloader_test):
            data_batch_dataNone = {k:v for k,v in data_batch_test.items() if v== None}
            data_batch_test = {k:v.to(args.device) for k,v in data_batch_test.items() if v!= None}
            data_batch_test.update(data_batch_dataNone)

            a=datetime.now() 

            test_scores_batch = model(data_batch_test)                   # (bs, num_nodes)
            b=datetime.now()   
            
            if i == 0:
                test_scores = test_scores_batch.cpu()
            else:
                test_scores = torch.cat((test_scores,test_scores_batch.cpu()), 0)
            c=datetime.now()   
            total_infer_time_ba += (b-a).total_seconds()
            total_infer_time_ca += (c-a).total_seconds()
    num_samples = test_scores.shape[0]
    if logtime:
        logging.info("num_samples:%s"%num_samples)
        logging.info("total_infer_time_ba:%s"%total_infer_time_ba)
        logging.info("total_infer_time_ba/num_samples:%f"%(total_infer_time_ba/num_samples))
        logging.info("num_samples/total_infer_time_ba:%f"%(num_samples/total_infer_time_ba))
        logging.info("total_infer_time_ca:%s"%total_infer_time_ca)
        logging.info("total_infer_time_ca/num_samples:%f"%(total_infer_time_ca/num_samples))
        logging.info("num_samples/total_infer_time_ca:%f"%(num_samples/total_infer_time_ca))
        logging.info("end test...")
    return test_scores.data.numpy() # test_scores.data.cpu().numpy()


def main_rank(args):
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
        model = BertResNet_2Inp(args, edges_np, node_id)
    elif args.model_name == "BertResNet_3Inp": 
        model = BertResNet_3Inp(args, edges_np, node_id)
    elif args.model_name == "ConvE": 
        model = ConvE(args, edges_np, node_id)
    else:
        raise "Model not implemented. Choose in [BertResNet_2Inp, BertResNet_3Inp, ConvE]"
    
    model = model.to(args.device)

    ################## training ##################
    if args.saved_model_name_rank == '':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max',factor = 0.5, patience = 2)
        criterion = torch.nn.BCELoss()
        
        loss_list = []
        mrr_list, mr_list = [], []
        hit10_list, hit30_list, hit50_list = [], [], []

        best_MR = 20000
        best_MRR = 0
        best_epoch = 0
        count = 0
        valid_perf = None
        valid_ranks = None
        valid_ranked_cands = None
        for epoch in range(args.n_epochs):
            model.train()
            if count >= args.early_stop: break
            epoch_loss = 0                                        
            
            for i,data_batch_train in enumerate(dataloader_train):
                data_batch_dataNone = {k:v for k,v in data_batch_train.items() if v== None}
                data_batch_train = {k:v.to(args.device) for k,v in data_batch_train.items() if v!= None}
                data_batch_train.update(data_batch_dataNone)
                
                if args.model_name in ["ConvE" ] and data_batch_train['labels'].shape[0] == 1:
                    continue

                optimizer.zero_grad()
                train_scores = model(data_batch_train)
                
                train_scores = torch.sigmoid(train_scores)
                loss = criterion(train_scores, data_batch_train['labels'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                epoch_loss += (loss.data).cpu().numpy()
            logging.info("epoch {}/{} total epochs, epoch_loss: {}".format(epoch+1,args.n_epochs,epoch_loss/i+1))
            

            if (epoch + 1)%args.eval_epoch==0:
                valid_scores = testmodel(model, dataloader_valid, args)
                perf, ranks, ranked_cands = evaluate_perf(basedata.valid_trips, valid_scores, args, basedata, K=10)
                logging.info("MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (perf["mrr"], perf["mr"], perf["hits@"][1], perf["hits@"][3], perf["hits@"][10], perf["hits@"][30], perf["hits@"][50]))
                perf['epoch'] = epoch+1
                MR, MRR= perf['mr'], perf['mrr']
                mrr_list.append(MRR)
                mr_list.append(MR)
                hit10_list.append(perf["hits@"][10])
                hit30_list.append(perf["hits@"][30])
                hit50_list.append(perf["hits@"][50])
                if MRR>best_MRR or MR<best_MR:
                    count = 0
                    if MRR>best_MRR: best_MRR = MRR
                    if MR<best_MR: best_MR = MR
                    valid_perf = perf
                    valid_ranks = ranks
                    valid_ranked_cands = ranked_cands
                    best_epoch = epoch + 1
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, args.model_path)
                else:
                    count+=1
                logging.info("Best Valid MRR: {}, Best Valid MR: {}, Best Epoch: {}".format(best_MRR,best_MR,best_epoch))
                scheduler.step(best_epoch)
            loss_list.append(epoch_loss)
    
        # plot
        if args.n_epochs > 2:
            plot_perf(loss_list, mrr_list, mr_list, hit10_list, hit30_list, hit50_list, args.plots_path)
    
        # save valid ranks and predictions
        if args.n_epochs >= args.eval_epoch:
            save_ranks(basedata.valid_trips, valid_ranks, args.ranks_path % "valid")
            save_preds(basedata.valid_trips, valid_ranked_cands, basedata, args.preds_path_npy % "valid", args.preds_path_txt % "valid")
    

    # test
    logging.info("------- Test Set Evaluation -------")
    checkpoint = torch.load(args.model_path)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    test_scores = testmodel(model, dataloader_test, args, logtime=True)
    
    test_perf, test_ranks, test_ranked_cands = evaluate_perf(basedata.test_trips, test_scores, args, basedata, K=10)
    # save_ranks(basedata.test_trips, test_ranks, args.ranks_path % "test")
    # save_preds(basedata.test_trips, test_ranked_cands, basedata, args.preds_path_npy % "test", args.preds_path_txt % "test")
    
    if args.saved_model_name_rank == '':
        test_perf['epoch'] = best_epoch 
    with open(args.perfs_path, 'w') as fout:
        json.dump({'test':test_perf, 'valid':valid_perf}, fout)
    logging.info("# MRR:%6f,  MR:%6f,  hits@1:%6f,  hits@3:%6f,  hits@10:%6f,  hits@30:%6f,  hits@50:%6f" % (test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))
  
    print("over")
    
    file_path = 'results_' + args.dataset +'.txt'
    with open(file_path, mode='a', encoding='utf-8') as file_obj:
        file_obj.write("%s\tMRR:%6f\tMR:%6f\thits@1:%6f\thits@3:%6f\thits@10:%6f\thits@30:%6f\thits@50:%6f\n" % (args.save_dir_name, test_perf["mrr"], test_perf["mr"], test_perf["hits@"][1], test_perf["hits@"][3], test_perf["hits@"][10], test_perf["hits@"][30], test_perf["hits@"][50]))



 
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

    main_rank(args)
