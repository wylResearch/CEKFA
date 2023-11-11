import os
import sys
import time
import uuid
import getpass
import argparse
import json
import logging
from configparser import ConfigParser

from lama.options import __add_bert_args, __add_roberta_args

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='CEKFA for OpenKG')

    parser.add_argument('--data_path', dest='data_path', default='./dataset/', 
                        help='directory path of KG datasets')       # TODO
    parser.add_argument('--dataset', dest='dataset', default='ReVerb20K', help='Dataset Choice')   
    
    parser.add_argument('-save', '--save_dir_name', default=None, type=str)

    parser.add_argument('--cuda', type=int, default=0, help='GPU id')
    
    parser.add_argument("--train_proportion", type=float, default=1.0)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--model_name', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--num_nodes', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_rels', type=int, default=0, help='DO NOT MANUALLY SET')
    
    ### bert init
    parser.add_argument("--bert_init",   default=False,  action='store_true', help="")
    parser.add_argument("--fc_bert_init",   default=False,  action='store_true', help="")
    parser.add_argument('--nodes_emb_file', type=str, default='', help='') 
    parser.add_argument('--rels_emb_file', type=str, default='', help='')   

    parser.add_argument('--dp', type=float, default=0., help='')

    ### NP 
    parser.add_argument('--np_neigh_mth', dest='np_neigh_mth', default=None, choices=[None, 'LAN'], help='wether to do canonicalization of np')

    ### RP 
    parser.add_argument('--rp_neigh_mth', dest='rp_neigh_mth', default=None, choices=[None, 'Local'], help='wether to do canonicalization of rp')
    parser.add_argument('--rpneighs_filename', dest='rpneighs_filename', default='new_sbert_ambv2_ENT_rpneighs', help='file name of similar rps.')
    parser.add_argument('--maxnum_rpneighs', dest='maxnum_rpneighs', default=5, type=int, help='max number of similar rps for each rp')
    parser.add_argument('--thresholds_rp', dest='thresholds_rp', default=0.8, type=float, help='thresholds')
 
    # retrieval
    parser.add_argument('--retrieval_basedir', dest='retrieval_basedir', default='retrieval', type=str, help=' ')
    parser.add_argument('--retrieval_model', dest='retrieval_model', default='/opt/data/private/PretrainedBert/sentence-transformers/all-mpnet-base-v2', type=str, 
                                        help='name or path of the retrieval model (SBert)') # TODO
    parser.add_argument('--retrieval_dirname', dest='retrieval_dirname', default='ambv2', type=str, help='name of the directory used to save the search results.')
    parser.add_argument('--retrieval_version', dest='retrieval_version', default='v2')
    parser.add_argument('--retrieval_Top_K', dest='retrieval_Top_K', default=10, type=int, help='retrieval Top K similar sentences for a query')

    # reranking    # TODO    /home/bdyw/data/pretrained_models/distilbert-base-uncased
    parser.add_argument('--finetune_rerank_model_dir', dest='finetune_rerank_model_dir', default='/opt/data/private/PretrainedBert/distilbert-base-uncased', type=str, help='the fine-tuning model dir')
    parser.add_argument('--finetune_rerank_model_name', dest='finetune_rerank_model_name', default='distilbert-base-uncased', type=str, help='the fine-tuning model name')
    
    parser.add_argument('--saved_model_path_rerank', dest='saved_model_path_rerank', default='', type=str, help='path of the saved rerank model')
    parser.add_argument('--rerank_Top_K', dest='rerank_Top_K', default=10, type=int, help='rerank top K predicted candidates.')
    parser.add_argument('--rerank_training_neg_K', dest='rerank_training_neg_K', default=10, type=int, help='train rerank network using predicted top K negtive tails')
    parser.add_argument('--rerank_bert_mth', dest='rerank_bert_mth', default='CLS', type=str, choices=['CLS', 'SUM', 'CLS_new'], help='method for bert rerank')
    parser.add_argument('--lr_finetune', dest='lr_finetune', default=0.001, type=float, help='learning rate for fine-tune rerank bert.')
    parser.add_argument("--omega", type=float, default=-1.0, help="weights for score of first stage.")

    ###
    parser.add_argument("--analysis",   default=False,  action='store_true', help="use this flag to analyse existing models")
    parser.add_argument('--Hits',        dest='Hits',         default= [1,3,10,30,50],           help='Choice of n in Hits@n')
    parser.add_argument('--seed', dest='seed',default=42, type=int, help='seed')

    __add_bert_args(parser)
    __add_roberta_args(parser)
    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model_name = argparse_dict['model_name']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    

def set_params(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    # input data dir
    args.data_path = os.path.join(args.data_path, args.dataset)
    
    # output data dir
    if args.np_neigh_mth == 'LAN':
        prefix = 'npLAN_'
    else:
        prefix = ''
    if args.rp_neigh_mth == 'Local':
        prefix = prefix + 'rp' + args.rp_neigh_mth + '_Num' + str(args.maxnum_rpneighs)
    else:
        prefix = prefix + 'rp_Num0'
    if args.bert_init:
        prefix = prefix+'_bertinit'
        if args.fc_bert_init:
            prefix = prefix+'_fc'
    if args.init_checkpoint is not None:
        args.save_path = args.init_checkpoint
    else:
        args.save_dir_name = '%s_%.1f_%s_%s_%s_%s_%s'%(args.model_name, args.train_proportion, args.hidden_dim, args.gamma, args.learning_rate, prefix, time.strftime("%Y-%m-%d")) if args.save_dir_name is None else args.save_dir_name
        args.save_path = 'results/%s/%s'%(args.dataset, args.save_dir_name) 
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # canonical rps directory
    args.rpneighs_dir = './files_supporting/%s/%s'%(args.dataset, 'canonical_rps')
    if not os.path.exists(os.path.abspath(args.rpneighs_dir)):
        os.makedirs(os.path.abspath(args.rpneighs_dir))
    # canonical triples directory
    args.retrieval_basedir = './files_supporting/%s/%s'%(args.dataset, 'canonical_triples')
    if not os.path.exists(os.path.abspath(args.retrieval_basedir)):
        os.makedirs(os.path.abspath(args.retrieval_basedir))

    train_file = '/train_trip_%.1f.txt'%args.train_proportion if args.train_proportion < 1 else '/train_trip.txt'
    
    # set data files
    args.data_files = {
        'ent2id_path'       : args.data_path + '/ent2id.txt',
        'rel2id_path'       : args.data_path + '/rel2id.txt',
        'train_trip_path'   : args.data_path + train_file,
        'test_trip_path'    : args.data_path + '/test_trip.txt',
        'valid_trip_path'   : args.data_path + '/valid_trip.txt',
        'gold_npclust_path' : args.data_path + '/gold_npclust.txt',
        'cesi_npclust_path' : args.data_path + '/cesi_npclust.txt',
        'cesi_rpneighs_path': args.data_path + '/cesi_rpclust.txt',
        'rpneighs_filepath' : args.rpneighs_dir + '/' + args.rpneighs_filename + '.txt',      
    }
    
    # using bert init
    if args.bert_init:
        if args.nodes_emb_file == '' or args.rels_emb_file == '':
            bert_init_emb_path = './files_supporting/' + args.dataset + '/bert_init_emb'
            if not os.path.exists(os.path.abspath(bert_init_emb_path)):
                os.makedirs(os.path.abspath(bert_init_emb_path))
            if args.nodes_emb_file == '':
                args.nodes_emb_file = os.path.join(bert_init_emb_path, 'Nodes_init_emb_' + args.dataset + '_bert-base-uncased.npy')
            if args.rels_emb_file == '':
                args.rels_emb_file = os.path.join(bert_init_emb_path, 'Rels_init_emb_' + args.dataset + '_bert-base-uncased_wo_rev.npy') 
        else:
            assert os.path.exists(args.nodes_emb_file), "nodes_emb_file not exists: " + args.nodes_emb_file
            assert os.path.exists(args.rels_emb_file), "rels_emb_file not exists: " + args.rels_emb_file
        args.models, args.bert_model_name = "bert", "bert-base-uncased"
        args.bert_dim = 768
    
    if args.saved_model_path_rerank != '' and args.omega < 0:
        raise ValueError("Please set valid value for omega")
    
    args.nfeats = args.hidden_dim
    args.n_epochs = args.max_steps
    args.reverse = False 
    return args


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')
    
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w',
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


