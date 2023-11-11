import os
import sys
import time
import uuid
import getpass
import argparse
import logging
from configparser import ConfigParser

from lama.options import __add_bert_args, __add_roberta_args

def getParser():
    parser = argparse.ArgumentParser(description='CEKFA for OpenKG')

    ### 
    parser.add_argument('--data_path', dest='data_path', default='./dataset/', help='directory path of KG datasets')        # TODO  ./
    parser.add_argument('--dataset', dest='dataset', default='ReVerb20K', help='Dataset Choice')
    
    parser.add_argument('-save', '--save_dir_name', default=None, type=str)

    parser.add_argument('--cuda', type=int, default=0, help='GPU id')

    parser.add_argument('--seed', dest='seed',default=42, type=int, help='seed')

    parser.add_argument("--train_proportion", type=float, default=1.0)


    ### model
    parser.add_argument('--saved_model_name_rank', dest='saved_model_name_rank', default='', type=str, help='directory name of the saved rank model')

    parser.add_argument('--model_name', dest='model_name', default='', help='model name')

    parser.add_argument('--bert_init',     dest='bert_init',     default=False,  action='store_true', help='for ConvE')
    parser.add_argument('--nodes_emb_file', type=str, default='', help='') 
    parser.add_argument('--rels_emb_file', type=str, default='', help='')

    ### NP 
    parser.add_argument('--np_neigh_mth', dest='np_neigh_mth', default=None, choices=[None, 'LAN'], help='wether to do canonicalization of np')

    ### RP 
    parser.add_argument('--rp_neigh_mth', dest='rp_neigh_mth', default=None, choices=[None, 'Global', 'Local'], help='wether to do canonicalization of rp')
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
    parser.add_argument('--do_rerank',   dest='do_rerank',   default=False,  action='store_true', help='')
    # PLM model
    parser.add_argument('--finetune_rerank_model_dir', dest='finetune_rerank_model_dir', default='', type=str, help='the fine-tuning model dir')
    parser.add_argument('--finetune_rerank_model_name', dest='finetune_rerank_model_name', default='distilbert-base-uncased', type=str, help='the fine-tuning model name')
    
    parser.add_argument('--saved_model_path_rerank', dest='saved_model_path_rerank', default='', type=str, help='path of the saved rerank model')
    parser.add_argument('--rerank_Top_K', dest='rerank_Top_K', default=10, type=int, help='rerank top K predicted candidates.')
    parser.add_argument('--rerank_training_neg_K', dest='rerank_training_neg_K', default=10, type=int, help='train rerank network using predicted top K negtive tails')
    parser.add_argument('--rerank_bert_mth', dest='rerank_bert_mth', default='CLS', type=str, choices=['CLS', 'SUM', 'CLS_new'], help='method for bert rerank')
    parser.add_argument('--lr_finetune', dest='lr_finetune', default=0.001, type=float, help='learning rate for fine-tune rerank bert.')
    parser.add_argument("--omega", type=float, default=-1.0, help="weights for score of first stage.")

    # # Clustering hyper-parameters
    # parser.add_argument('-linkage', dest='linkage', default='complete', choices=['complete', 'single', 'avergage'], help='HAC linkage criterion')
    # parser.add_argument('-thresh_val', dest='thresh_val', default=.4239, type=float, help='Threshold for clustering')
    # parser.add_argument('-metric', dest='metric', default='cosine', help='Metric for calculating distance between embeddings')
    # parser.add_argument('-num_canopy', dest='num_canopy', default=1, type=int, help='Number of caponies while clustering')

    ###  ResNet
    parser.add_argument("--reshape_len", type=int, default=5, help="Side length for deep convolutional models")
    parser.add_argument("--resnet_num_blocks", type=int, default=2, help="Number of resnet blocks")
    parser.add_argument("--resnet_block_depth", type=int, default=3, help="Depth of each resnet block")
    parser.add_argument("--resnet_dropout", type=float, default=0.3, help="dropout probability")
    parser.add_argument("--input_dropout", type=float, default=0.2, help="input dropout")
    parser.add_argument("--feature_map_dropout", type=float, default=0.2, help="feature map dropout")
        

    #### Hyper-parameters
    parser.add_argument('--nfeats',      dest='nfeats',       default=768,   type=int,       help='Embedding Dimensions')
    # parser.add_argument('--num_layers',  dest='num_layers',   default=1,     type=int,       help='No. of layers in encoder network')
    # parser.add_argument('--bidirectional',  dest='bidirectional',   default=True,     type=bool,       help='type of encoder network')
    # parser.add_argument('--poolType',    dest='poolType',     default='last',choices=['last','max','mean'], help='pooling operation for encoder network')
    parser.add_argument('--dropout',     dest='dropout',      default=0.5,   type=float,     help='Dropout')
    # parser.add_argument('--reg_param',   dest='reg_param',    default=0.0,   type=float,     help='regularization parameter')
    parser.add_argument('-lr',          dest='learning_rate',           default=0.0001, type=float,     help='learning rate')
    # parser.add_argument('--p_norm',      dest='p_norm',       default=1,     type=int,       help='TransE scoring function')
    parser.add_argument('--batch_size',  dest='batch_size',   default=128,   type=int,       help='batch size for training')
    # 
    parser.add_argument('--num_workers',  dest='num_workers',   default=0,   type=int,       help='number workers for dataloader')

    # parser.add_argument('--neg_samples', dest='neg_samples',  default=10,    type=int,       help='No of Negative Samples for TransE')
    parser.add_argument('--n_epochs',    dest='n_epochs',     default=500,   type=int,       help='maximum no. of epochs')
    parser.add_argument('--grad_norm',   dest='grad_norm',    default=1.0,   type=float,     help='gradient clipping')
    parser.add_argument('--eval_epoch',  dest='eval_epoch',   default=5,     type=int,       help='Interval for evaluating on validation dataset')
    parser.add_argument('--Hits',        dest='Hits',         default= [1,3,10,30,50],           help='Choice of n in Hits@n')
    parser.add_argument('--early_stop',  dest='early_stop',   default=10,    type=int,       help='Stopping training after validation performance stops improving')

    # parser.add_argument('--lamafile',    dest='lamafile',    type=str,      default="",     help='file containing lama model')
    # parser.add_argument('--lamaweight',  dest='lamaweight',  type=float,    default=0.0,    help='weight for LAMA models score')
    # parser.add_argument('--careweight',  dest='careweight',  type=float,    default=1.0,    help='weight for CaRE models score')
    parser.add_argument('--reverse',     dest='reverse',     default=True,  action='store_true', help='whether to add inverse relation edges')
    parser.add_argument('--eval-test',   dest='eval_test',   default=False,  action='store_true', help='flag to evaluate on test split instead of valid split')

    # logging options
    parser.add_argument("--name",           type=str,   default='', help="Set filename for saving or restoring models")



    # Bert options
    parser.add_argument("--language-models", "--lm", dest="models",  default='bert', help="comma separated list of language models")

    __add_bert_args(parser)
    __add_roberta_args(parser)
    return parser

def set_params(args):
    # input data dir
    args.data_path = os.path.join(args.data_path, args.dataset) 

    # output data dir
    if args.np_neigh_mth == 'LAN':
        prefix = 'npLAN_'
    else:
        prefix = ''
    if args.rp_neigh_mth in ['Local', 'Global']:
        prefix = prefix + 'rp' + args.rp_neigh_mth + '_Num' + str(args.maxnum_rpneighs)
    else:
        prefix = prefix + 'rp_Num0'
    if args.bert_init:
        prefix = prefix+'_bertinit'
    
    if args.saved_model_name_rank != '':
        args.save_path = args.saved_model_name_rank
    else:
        args.save_dir_name = '%s_%.1f_%s_%s_%s_%s'%(args.model_name, args.train_proportion, args.nfeats, args.learning_rate, prefix, time.strftime("%Y-%m-%d")) if args.save_dir_name is None else args.save_dir_name
        args.save_path = 'results/%s/%s'%(args.dataset, args.save_dir_name) 
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.name == '':
        args.name = args.save_dir_name
    
    # canonical rps directory
    args.rpneighs_dir = './files_supporting/%s/%s'%(args.dataset, 'canonical_rps')
    if not os.path.exists(os.path.abspath(args.rpneighs_dir)):
        os.makedirs(os.path.abspath(args.rpneighs_dir))
    # canonical triples directory
    args.retrieval_basedir = './files_supporting/%s/%s'%(args.dataset, 'canonical_triples')
    if not os.path.exists(os.path.abspath(args.retrieval_basedir)):
        os.makedirs(os.path.abspath(args.retrieval_basedir))
    
    args.model_path = os.path.join(args.save_path, 'model.pth')
    args.outdata_path = os.path.join(args.save_path, 'data.pkl')
    args.preds_path_txt = os.path.join(args.save_path, f'preds_%s.txt')
    args.preds_path_npy = os.path.join(args.save_path, f'preds_%s.npy')
    args.ranks_path = os.path.join(args.save_path, f'ranks_%s.npy')
    args.perfs_path = os.path.join(args.save_path, 'perfs.json')
    args.plots_path = os.path.join(args.save_path, 'loss.png')

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
                args.rels_emb_file = os.path.join(bert_init_emb_path, 'Rels_init_emb_' + args.dataset + '_bert-base-uncased_w_rev.npy') 
        else:
            assert os.path.exists(args.nodes_emb_file), "nodes_emb_file not exists: " + args.nodes_emb_file
            assert os.path.exists(args.rels_emb_file), "rels_emb_file not exists: " + args.rels_emb_file
        args.models, args.bert_model_name = "bert", "bert-base-uncased"
        args.bert_dim = 768    
    
    if args.saved_model_path_rerank != '' and args.omega < 0:
        raise ValueError("Please set valid value for omega")

    # set bert dimensions
    if args.models in ['bert']:
        if args.bert_model_name.startswith('bert-base'):
            args.bert_dim = 768
        else:
            args.bert_dim = 1024
    elif args.models in ['roberta']:
        if args.roberta_model_dir is None:
            print("please specify --rmd (roberta model directory)")
        elif args.roberta_model_dir.endswith('base'):
            args.bert_dim = 768
        else:
            args.bert_dim = 1024

    return args





def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if not args.do_rerank:
        if args.saved_model_name_rank == '':
            log_file = os.path.join(args.save_path, 'train.log')
        else:
            log_file = os.path.join(args.save_path, 'test.log')
    else:
        if args.saved_model_path_rerank == '':
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


