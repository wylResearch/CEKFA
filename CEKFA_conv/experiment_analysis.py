
 #### Import all the supporting classes
from ipaddress import _BaseAddress
import random
import os
import numpy as np
from sklearn import neighbors
import torch
import json
from sklearn.manifold import TSNE
from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter

from parse_args import *
from data import *
from evaluate import *


from models import *




# color_dict = {1:'black', 2:'indigo', 3: 'black', 4:'darkgoldenrod', 
#                     5:'forestgreen', 6:'royalblue', 7:'dimgray', 8:'firebrick'} # label2color
color_dict = {  2:'#fe4b03', # orange
                4:'#742802', # brown
                5:'#018529', # green
                6:'#021bf9', # blue
                7:'gray', 
                8:'#b0054b',   # red
                1:'black',3: 'black'} # label2color
# 1,3 没有
def print_dataset(basedata):
    # entid = basedata.ent2id['humanism']
    # ents = basedata.true_clusts[entid]
    # ents = [basedata.id2ent[id] for id in ents]

    # filename = '/opt/data/private/KGuseBert/CaRE-master/Data/ReVerb20K/train_trip_str.txt'
    # for line in open(filename, "r"):
    #     if "gender" in line:
    #         print(line.strip())

    filename = '/opt/data/private/KGuseBert/CaRE-master/Data/ReVerb20K/train_trip.txt'
    for line in open(filename, "r"):
        line = [int(x) for x in line.strip().split()]
        if len(line) == 3:
            h = basedata.id2ent[line[0]]
            r = basedata.id2rel[line[1]]
            t = basedata.id2ent[line[2]]
            sen = h + "  -  " +  r + "  -  " + t
            # if "gender" in h:
            #     t_ = basedata.true_clusts[line[2]]
            #     if len(t_) > 1:
            #         print(sen)
            # elif "gender" in t:
            #     h_ = basedata.true_clusts[line[0]]
            #     if len(h_) > 1:
            #         print(sen)
            if "gates" in sen:
                print(sen)

    # filename = '/opt/data/private/KGuseBert/RPs-wyl-v9-clean/results/rpneighs/ReVerb20K/sbert_ambv2_AB_rpneighs.txt'
    # for line in open(filename, "r"):
    #     line = [int(x) for x in line.strip().split()]
    #     if basedata.id2rel[line[0]] ==  "be central to":
    #         for x in line[:10]:
    #             print(basedata.id2rel[x])

    # for input in ['life', 'child', 'identity', 'bill clinton', 'point', 'gender justice']:
    #     entid = basedata.ent2id[input]
    #     ents = basedata.true_clusts[entid]
    #     ents = [basedata.id2ent[id] for id in ents]
    #     print('---')
    #     for ent in ents:
    #         print(ent)


def process_preds(file):
    fw = open('rerank_analysis.txt', "w")
    flag = False
    lines = open(file, "r").readlines()
    for i in range(len(lines)):
        if i%9 == 0 and "--InTopK" in lines[i]:
            tail_target = lines[i].split(",")[-1].split("]")[0][:-1]
            flag = True
            line_triple = lines[i]
        elif i%9 != 0 and flag == True:
            if i%9 == 1: 
                candis_raw = [item.strip() for item in lines[i].replace("\t","").split(';')[:10]]
                candis_new =[item.strip() for item in lines[i+6].replace("\t","").split(';')[:10]]
                
                if tail_target in candis_raw:
                    idx_raw = candis_raw.index(tail_target)
                    idx_new = candis_new.index(tail_target)

                    if idx_new < idx_raw:
                        fw.write('-----%d\n'% (idx_raw-idx_new))
                        fw.write(line_triple)
                        fw.write(lines[i])
                        fw.write(lines[i+6])
                flag = False
            else:
                continue
        else:
            continue
    fw.close()

def process_rp_neighbour_line(line):
    # eg. : 16754	was a different name for:	19326	was another name for	0.9703336954116821;	9476	was an alternate name for	0.9610127806663513;	16870	had another name for	0.9553003907203674;	20228	was previously called	0.9524335265159607;	13786	was known previously as	0.9514162540435791;	8974	was earlier called	0.9510113596916199;	15063	was called	0.9473333954811096;	19175	was originally named	0.9463474154472351;	13383	was also named	0.9462819695472717;	16604	was originally called	0.9454656839370728;	17735	was earlier called as	0.9442480206489563;	4356	was then renamed	0.9424322247505188;	16094	was previously known as	0.9424027800559998;	12059	was also a name for	0.9416919350624084;	12672	was once called	0.9409686326980591;	21090	was later re named	0.9409130215644836;	9516	was later renamed to	0.9407490491867065;	6742	was a designation for	0.9402023553848267;	9741	was the real name of	0.9400752186775208;	21500	was later changed to	0.9389989972114563
    idx_list = []
    str_list = []
    line = line.strip().split(':')
    
    if len(line) == 2:
        idx, rp = line[0].split('\t')
        idx_list.append(int(idx))
        str_list.append(rp)
        neighs = line[1].split(';')
    else:
        raise Exception

    for neigh in neighs:
        item = neigh.split('\t')
        idx_list.append(int(item[1]))
        str_list.append(item[2])
    return idx_list, str_list
 
def write_emb_data_ours(args, basedata):
    # 加载模型获得嵌入层的参数
    args.saved_model_name_rank = 'BertResNet_2Inp_Local_Max10_ReVerb45K_bbu_new_ambv2_ENT_14_04_2022_17:40:04.model.pth'  
    
    saved_model_path_rank = os.path.join(args.save_path, args.saved_model_name_rank)        # TODO
    if os.path.exists(saved_model_path_rank):
        edges_np = torch.tensor(basedata.edges, dtype=torch.long).to(args.device)     
        node_id = torch.arange(0, basedata.num_nodes, dtype=torch.long).to(args.device)
        if args.model_name == "BertResNet_2Inp": 
            model_1 = BertResNet_2Inp(args, edges_np, node_id)
        elif args.model_name == "BertResNet_3Inp": 
            model_1 = BertResNet_3Inp(args, edges_np, node_id)
        else:
            raise "Model not implemented. Choose in [BertResNet_2Inp, BertResNet_3Inp]"

        model_1 = model_1.to(args.device)
        
        checkpoint = torch.load(saved_model_path_rank)
        model_1.eval()
        model_1.load_state_dict(checkpoint['state_dict'])

        rp_emb = model_1.rp_embeddings.weight
        rp_emb = rp_emb.detach().numpy()
        np.save('rp_emb_ourbase_45k.npy', rp_emb)
    else:
        raise Exception

    return     

def get_emb_data_ours(rp_idx_list):
    rels_emb_file = 'rp_emb_ourbase_45k.npy'

    # 加载模型获得嵌入层的参数
    if os.path.exists(rels_emb_file):
        rp_emb = np.load(rels_emb_file)
        rp_emb = torch.FloatTensor(rp_emb)
    else:
        raise Exception

    # 根据 rp_idx_list 选出我们需要的 rp 的 emb
    data_emb = rp_emb[rp_idx_list]
    return data_emb.detach().numpy()      # n_samples, n_features = data.shape

def get_emb_data_care(rp_idx_list):
    rels_emb_file = '/opt/data/private/KGuseBert/CaRE-master/rp_emb_care_45k.npy'

    # 加载模型获得嵌入层的参数
    if os.path.exists(rels_emb_file):
        rp_emb = np.load(rels_emb_file)
        rp_emb = torch.FloatTensor(rp_emb)
    else:
        raise Exception

    # 根据 rp_idx_list 选出我们需要的 rp 的 emb
    data_emb = rp_emb[rp_idx_list]
    return data_emb.detach().numpy()      # n_samples, n_features = data.shape

def get_emb_data_bertinit(rp_idx_list):
    rels_emb_file = '/opt/data/private/KGuseBert/RPs-wyl-v3/bert_init_emb/Rels_init_emb_ReVerb45K_bert-base-uncased.npy'

    # 加载模型获得嵌入层的参数
    if os.path.exists(rels_emb_file):
        rp_emb = np.load(rels_emb_file)
        rp_emb = torch.FloatTensor(rp_emb)
    else:
        raise Exception

    # 根据 rp_idx_list 选出我们需要的 rp 的 emb
    data_emb = rp_emb[rp_idx_list]
    return data_emb.detach().numpy()      # n_samples, n_features = data.shape

def get_emb_data_okgit(rp_idx_list):
    rels_emb_file = '/opt/data/private/KGuseBert/OKGIT-wyl/rp_emb_okgit_45k.npy'

    # 加载模型获得嵌入层的参数
    if os.path.exists(rels_emb_file):
        rp_emb = np.load(rels_emb_file)
        rp_emb = torch.FloatTensor(rp_emb)
    else:
        raise Exception

    # 根据 rp_idx_list 选出我们需要的 rp 的 emb
    data_emb = rp_emb[rp_idx_list]
    return data_emb.detach().numpy()      # n_samples, n_features = data.shape

def main_tsne(data):
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=35, init='pca', random_state=0)
    result = tsne.fit_transform(data)       # 将输入的高维数据降维了
    return result
   
# 总体对比
def plot_embedding(data, label, rp_idx_list, rp_str_list, show_idx_list, title, wfilename):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min) 

    data[:,0] = data[:,0] * 0.97

    fig = plt.figure()
    ax = plt.subplot(111)
    
    #画散点图
    data_idx = 0
    for i in range(len(label)):
        for j in range(len(label[i])):
            label_ij = label[i][j]
            if rp_idx_list[data_idx] not in show_idx_list:
                color='white' # white, k
                # color = plt.cm.prism(label_ij / 15.)  # twilight
                # ax.scatter(data[data_idx, 0], data[data_idx, 1], c=color)
                # plt.text(data[data_idx, 0], data[data_idx, 1], rp_str_list[i][j],
                #         color=color,
                #         fontdict={'weight': 'light', 'size': 5})
            else:
                # color = plt.cm.prism(label_ij / 15.)  # twilight
                color = color_dict[label_ij]
                ax.scatter(data[data_idx, 0], data[data_idx, 1], c=color)
                plt.text(data[data_idx, 0], data[data_idx, 1]-0.01, ' '+rp_str_list[i][j],
                        color=color,
                        fontdict={'weight': 'light', 'size': 13})
            data_idx += 1
    ax.scatter(1.2, 1, c='k')
    
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    plt.savefig(wfilename + '.pdf')
    return fig

# 微调各个点的标签位置
def plot_embedding_ours(data, label, rp_idx_list, rp_str_list, show_idx_list, title, wfilename, subplot=111):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min) 

    data[:,0] = data[:,0] * 0.94

    if subplot == 111:
        fig = plt.figure()
    ax = plt.subplot(subplot)
    
    #画散点图
    data_idx = 0
    for i in range(len(label)):
        for j in range(len(label[i])):
            label_ij = label[i][j]
            if rp_idx_list[data_idx] in show_idx_list:
                color = color_dict[label_ij]
                # color = plt.cm.prism(label_ij / 15.)  # twilight
                ax.scatter(data[data_idx, 0], data[data_idx, 1], c=color)
                if rp_idx_list[data_idx] in [15412, 18115, 7916]:   # 往上移动一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]+0.01, ' '+rp_str_list[i][j],  # ' '+str(rp_idx_list[data_idx])+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [768, 16367, 7945, 17451]:   # 往下移动一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]-0.02, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [18301]:   # 往上移动多一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]+0.02, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                else:
                    plt.text(data[data_idx, 0], data[data_idx, 1]-0.01, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
            data_idx += 1
    ax.scatter(1.3, 1, c='white')
    
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    if subplot == 111:
        plt.savefig(wfilename + '.pdf')
        return fig
    else:
        return 

# 微调各个点的标签位置
def plot_embedding_care(data, label, rp_idx_list, rp_str_list, show_idx_list, title, wfilename, subplot=111):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min) 

    data[:,0] = data[:,0] * 0.95

    if subplot == 111:
        fig = plt.figure()
    ax = plt.subplot(subplot)
    
    #画散点图
    data_idx = 0
    for i in range(len(label)):
        for j in range(len(label[i])):
            label_ij = label[i][j]
            if rp_idx_list[data_idx] in show_idx_list:
                color = color_dict[label_ij]
                # color = plt.cm.prism(label_ij / 15.)  # twilight
                ax.scatter(data[data_idx, 0], data[data_idx, 1], c=color)
                if rp_idx_list[data_idx] in [3748, 12072]:   # 往上移动一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]+0.01, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [17451]:   # 往下移动一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]-0.02, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [16367, 18115]:   # 往上往左移动
                    plt.text(data[data_idx, 0]-0.01, data[data_idx, 1]+0.02, rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [1241]:   # 往上往左移动多一点
                    plt.text(data[data_idx, 0]-0.07, data[data_idx, 1]+0.02, rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [7916, 1745, 15412]:   # 往下移动一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]-0.02, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [14785]:   # 往下往左一点
                    plt.text(data[data_idx, 0]-0.03, data[data_idx, 1]-0.06, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [8632]:   # 往下往左一点
                    plt.text(data[data_idx, 0]-0.03, data[data_idx, 1]-0.06, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [7945]:   # 往下移动多一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]-0.03, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                else:
                    plt.text(data[data_idx, 0], data[data_idx, 1]-0.01, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
            data_idx += 1
    ax.scatter(1.25, 1, c='white')
    
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    if subplot == 111:
        plt.savefig(wfilename + '.pdf')
        return fig
    else:
        return 

# 微调各个点的标签位置
def plot_embedding_okgit(data, label, rp_idx_list, rp_str_list, show_idx_list, title, wfilename, subplot=111):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min) 

    data[:,0] = data[:,0] * 0.95

    if subplot == 111:
        fig = plt.figure()
    ax = plt.subplot(subplot)
    
    #画散点图
    data_idx = 0
    for i in range(len(label)):
        for j in range(len(label[i])):
            label_ij = label[i][j]
            if rp_idx_list[data_idx] in show_idx_list:
                color = color_dict[label_ij]
                # color = plt.cm.prism(label_ij / 15.)  # twilight
                ax.scatter(data[data_idx, 0], data[data_idx, 1], c=color)
                if rp_idx_list[data_idx] in [1241, 18301]:   # 往上移动一点
                    plt.text(data[data_idx, 0], data[data_idx, 1]+0.01, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                elif rp_idx_list[data_idx] in [14785]:   # 往上移动一点
                    plt.text(data[data_idx, 0]-0.01, data[data_idx, 1]+0.01, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
                else:
                    plt.text(data[data_idx, 0], data[data_idx, 1]-0.01, ' '+rp_str_list[i][j],
                        color=color, fontdict={'weight': 'light', 'size': 13})
            data_idx += 1
    ax.scatter(1.2, 1, c='white')
    
    plt.xticks([])
    plt.yticks([])
    # plt.title(title)
    if subplot == 111:
        plt.savefig(wfilename + '.pdf')
        return fig
    else:
        return 

# 三个子图合并  显示效果很不好
def plot_merge(data_care, data_okgit, data_ours, label, rp_idx_list, rp_str_list, show_idx_list, wfilename):
    #### care
    fig = plt.figure(figsize=(12,3))
    plot_embedding_care(data_care, label, rp_idx_list, rp_str_list, show_idx_list, 'care', wfilename, subplot=131)
    plot_embedding_okgit(data_okgit, label, rp_idx_list, rp_str_list, show_idx_list, 'okgit', wfilename, subplot=132)
    plot_embedding_ours(data_ours, label, rp_idx_list, rp_str_list, show_idx_list, 'ours', wfilename, subplot=133)
    plt.savefig(wfilename + '.pdf')
    return 


#####################################
# 输出 np与rp 的稀疏度
def cal_raw_sparsity(basedata):
    np2freq = defaultdict(int)
    rp2freq = defaultdict(int)
    for np1,rp,np2 in basedata.train_trips_without_rev:
        np2freq[np1]+=1
        np2freq[np2]+=1
        rp2freq[rp]+=1
    
    np_set = [i for i in range(len(basedata.id2ent))]
    rp_set = [i for i in range((len(basedata.id2rel)-1)//2)]
    np_set_now = [np for np in np2freq.keys()]
    rp_set_now = [rp for rp in rp2freq.keys()]

    for np in set(np_set) - set(np_set_now):
        np2freq[np] = 0
    for rp in set(rp_set) - set(rp_set_now):
        rp2freq[rp] = 0
    
    assert len(np_set) == len(np2freq)
    assert len(rp_set) == len(rp2freq)

    npfreq_list= [v for k,v in np2freq.items()]
    rpfreq_list= [v for k,v in rp2freq.items()]
    npfreq_max, npfreq_min = max(npfreq_list), min(npfreq_list)
    rpfreq_max, rpfreq_min = max(rpfreq_list), min(rpfreq_list)

    rpfreq_counter = Counter(rpfreq_list)
    sparsity_eq_1 = dict(rpfreq_counter).get(rpfreq_min, -1)
    print('---npfreq_max:', npfreq_max, ',npfreq_min:', npfreq_min, ',npfreq_avg:', sum(npfreq_list)/len(npfreq_list))
    print('---rpfreq_max:', rpfreq_max, ',rpfreq_min:', rpfreq_min, ',rpfreq_avg:', sum(rpfreq_list)/len(rpfreq_list))
    print('---rp_sparsity=1, num:', sparsity_eq_1)

    np2sparsity = defaultdict(float)
    rp2sparsity = defaultdict(float)
    for np, freq in np2freq.items():
        np2sparsity[np] = 1- (freq - npfreq_min)/(npfreq_max - npfreq_min)
    for rp, freq in rp2freq.items():
        rp2sparsity[rp] = 1- (freq - rpfreq_min)/(rpfreq_max - rpfreq_min)
    print_hints_sparsity(np2sparsity, 'NP', len(np2freq))
    print_hints_sparsity(rp2sparsity, 'RP', len(rp2freq))
    
    return np2sparsity, rp2sparsity, np2freq, rp2freq
    
def cal_new_sparsity(basedata):
    np2freq = defaultdict(int)
    rp2freq = defaultdict(int)
    for np1,rp,np2 in basedata.train_trips_without_rev:
        np2freq[np1]+=1
        np2freq[np2]+=1
        rp2freq[rp]+=1
    
    np_set = [i for i in range(len(basedata.id2ent))]
    rp_set = [i for i in range((len(basedata.id2rel)-1)//2)]
    
    np_set_now = [np for np in np2freq.keys()]
    rp_set_now = [rp for rp in rp2freq.keys()]
    for np in set(np_set) - set(np_set_now):
        np2freq[np] = 0
    for rp in set(rp_set) - set(rp_set_now):
        rp2freq[rp] = 0
    
    np2freq_new = defaultdict(int)
    for nps in basedata.unique_clusts:
        freq_cluster = 0
        for np in nps:
            freq_cluster += np2freq[np]
        for np in nps:
            np2freq_new[np] = freq_cluster
    
    np_set_now = [np for np in np2freq_new.keys()]
    for np in set(np_set) - set(np_set_now):
        np2freq_new[np] = 0


    rp2freq_new = defaultdict(int)
    if basedata.args.dataset == 'ReVerb20K':
        topk = 5
    else:
        topk = 10 
    for rp in list(rp2freq.keys()):
        similar_rps = basedata.rps2neighs[rp][:topk]
        rp2freq_new[rp] = sum([rp2freq[r] for r in similar_rps]) + rp2freq[rp]


    assert len(np_set) == len(np2freq_new)
    assert len(rp_set) == len(rp2freq_new)
    
    npfreq_list= [v for k,v in np2freq_new.items()]
    rpfreq_list= [v for k,v in rp2freq_new.items()]
    npfreq_max, npfreq_min = max(npfreq_list), min(npfreq_list)
    rpfreq_max, rpfreq_min = max(rpfreq_list), min(rpfreq_list)

    rpfreq_counter = Counter(rpfreq_list)
    sparsity_eq_1 = dict(rpfreq_counter).get(rpfreq_min, -1)
    print('---npfreq_max:', npfreq_max, ',npfreq_min:', npfreq_min, ',npfreq_avg:', sum(npfreq_list)/len(npfreq_list))
    print('---rpfreq_max:', rpfreq_max, ',rpfreq_min:', rpfreq_min, ',rpfreq_avg:', sum(rpfreq_list)/len(rpfreq_list))
    print('---rp_sparsity=1, num:', sparsity_eq_1)

    np2sparsity = defaultdict(float)
    rp2sparsity = defaultdict(float)
    for np, freq in np2freq_new.items():
        np2sparsity[np] = 1- (freq - npfreq_min)/(npfreq_max - npfreq_min)
    for rp, freq in rp2freq_new.items():
        rp2sparsity[rp] = 1- (freq - rpfreq_min)/(rpfreq_max - rpfreq_min)
    print_hints_sparsity(np2sparsity, 'NP', len(np2freq))
    print_hints_sparsity(rp2sparsity, 'RP', len(rp2freq))
    
    return np2sparsity, rp2sparsity, np2freq_new, rp2freq_new

def cal_sparsity_relative(content2sparsity, content2sparsity_new, content):
    # 计算相对降低多少的稀疏度
    relative_list = []
    for cont in content2sparsity:
        sparsity1 = content2sparsity[cont]
        sparsity2 = content2sparsity_new[cont]
        if sparsity1 != 0:
            x = (sparsity1-sparsity2)/sparsity1
            relative_list.append(x)
    print('**********************', content, ' sparsity **********************')
    print(len(content2sparsity), len(relative_list))
    print('max:', max(relative_list), ', min:', min(relative_list), ', avg:', sum(relative_list)/len(relative_list)) 

def cal_frequency_relative(content2freq, content2freq_new, content):
    # 计算相对降低多少的稀疏度
    relative_list = []
    for cont in content2freq:
        freq1 = content2freq[cont]
        freq2 = content2freq_new[cont]
        if freq1 != 0:
            x = (freq2-freq1)/freq1
            relative_list.append(x)
    print('**********************', content, ' frequency **********************')
    print(len(content2freq), len(relative_list))
    print('max:', max(relative_list), ', min:', min(relative_list), ', avg:', sum(relative_list)/len(relative_list))
   


def print_hints_sparsity(content2sparsity, content, totalnum):
    bound_1 = np.arange(0, 0.9, 0.1)
    bound_2 = np.arange(0.90, 1.01, 0.01)
    bound = np.append(bound_1, bound_2)
    bound_num = [0 for i in range(len(bound))]
    for k,sparsity in content2sparsity.items():
        if sparsity < 0.9:
            idx = int(str(sparsity)[2])
        elif sparsity >= 0.9 and sparsity < 1:
            idx = int(str(sparsity)[3]) + 9
        elif sparsity == 1:
            idx = 19
        else:
            print(sparsity)
            raise Exception
        bound_num[idx] += 1

    assert sum(bound_num) == len(content2sparsity) == totalnum

    print('-----------------------')
    print(content, sum(bound_num))
    # for i in range(len(bound_num)):
    #     if i < len(bound_num) -1:
    #         print('[{:.2f},{:.2f})'.format(bound[i], bound[i+1]), end=" ")
    #     else:
    #         print('[{:.2f},{:.2f}]'.format(bound[i], 1)) 
    # for i in range(len(bound_num)):
    #     print('{:<11.2f}'.format(bound_num[i]), end=" ")
    bound_print = [0,9,14,18,19]
    for i, _ in enumerate(bound_print):
        bound_s = bound_print[i]
        if i < len(bound_print) -1:
            bound_e = bound_print[i+1]
            print('[{:.2f},{:.2f})'.format(bound[bound_s], bound[bound_e]), end=" ")
        else:
            bound_e = len(bound_num) - 1
            print('[{:.2f},{:.2f}]'.format(bound[bound_s], bound[bound_e]))
    for i, _ in enumerate(bound_print):
        bound_s = bound_print[i]
        if i < len(bound_print) -1:
            bound_e = bound_print[i+1]
        else:
            bound_e = len(bound_num)
        num = [bound_num[idx] for idx in range(bound_s, bound_e)]
        print('{:<11.2f}'.format(sum(num)), end=" ")
    print()
        
    # print('---[0.0,0.9):', sum(bound_num[:9]))  # , end=" ")        
    


def plot_sparsity(content2sparsity, wfilename):
    x = [v for k,v in content2sparsity.items()]
    plt.figure() #初始化一张图
    plt.hist(x, bins=100)  #直方图关键操作
    plt.grid(alpha=0.1,linestyle='-.', axis='x') #网格线，更好看 
    plt.xlabel('sparsity')  
    plt.ylabel('Number')  
    # plt.title(r'sparsity') 
    plt.savefig(wfilename)


def plot_frequency_bar(content2freq, wfilename):
    content_freq_list = [v for k,v in content2freq.items()]
    x = [i for i in content_freq_list if i <=50 ]
    bin_num = max(x)
    plt.figure() #初始化一张图
    plt.hist(x, bins=bin_num)  #直方图关键操作
    plt.grid(alpha=0.1,linestyle='-.', axis='x') #网格线，更好看 
    plt.xlabel('frequency')  
    plt.ylabel('Number')  
    # plt.title(r'sparsity') 
    plt.savefig(wfilename)


def plot_frequency_line(content2freq, content2freq_new, wfilename, content):
    content_freq_list = [v for k,v in content2freq.items()]
    content_freq_list_new = [v for k,v in content2freq_new.items()]
    plt.figure(figsize=(5,4))
    content_counter = dict(Counter(content_freq_list))
    content_counter_new = dict(Counter(content_freq_list_new))
    x = [i for i in range(0,21)]
    y_freq, y_freq_new = [], []
    for freq in x:
        y_freq.append(dict(content_counter).get(freq, 0))
        y_freq_new.append(dict(content_counter_new).get(freq, 0))
    plt.plot(x, y_freq, alpha=0.8, linewidth=2, label='w/o', marker='.', markersize=5, color='orange')  # #e17701
    plt.plot(x, y_freq_new, alpha=0.8, linewidth=2, label='w/', marker='.', markersize=5, color='blue')

    plt.xticks([i for i in range(0,21,5)],fontsize=7,fontproperties = 'Times New Roman')  #  todo    tick_label 里面的字符太长了
    plt.yticks(fontsize=7,fontproperties = 'Times New Roman')

    plt.xlabel('frequency', fontsize=8,fontproperties = 'Times New Roman')  
    plt.ylabel('Number of ' + content, fontsize=8,fontproperties = 'Times New Roman')  
    plt.legend() 
    plt.savefig(wfilename)





#####################################
# 根据尾np的度将测试集分组，输出不同组重排序的性能提升
def split_testset(basedata):
    np2freq = defaultdict(int)
    for np1,rp,np2 in basedata.train_trips_without_rev:
        np2freq[np1]+=1
        np2freq[np2]+=1
    triple2tailnpfreq = defaultdict(int)
    tailnpfreq2triple = defaultdict(list)
    for triple in basedata.test_trips:      
        np1,rp,np2 = triple
        triple_ = tuple(list(triple))
        triple2tailnpfreq[triple_] = np2freq[np2]
        tailnpfreq2triple[np2freq[np2]].append(triple)
    return tailnpfreq2triple, triple2tailnpfreq

class SplittestDataset(Dataset):
    def __init__(self, args, basedata, test_trips):
        self.args = args
        self.basedata = basedata
        self.test_pairs = [[pair[0], pair[1]] for pair in test_trips]

    def __getitem__(self, index):
        return self.test_pairs[index]

    def __len__(self):
        return len(self.test_pairs)
    
    def get_batch_Local_rps_test(self, raw_batch): 
        neigh_rels_id = []
        for i in range(len(raw_batch)):             
            h_id, r_id = raw_batch[i]    
            if r_id in self.basedata.rps2neighs:
                neighs = self.basedata.rps2neighs[r_id]
            else:
                neighs_rev = self.basedata.rps2neighs[self.basedata.inverse2rel_map[r_id]]
                neighs = [self.basedata.inverse2rel_map[n_rev] for n_rev in neighs_rev]
            neigh_rels_id.append(neighs)
        pairs_id = torch.LongTensor(raw_batch)
        neigh_rels_id = torch.LongTensor(neigh_rels_id)
        return {'pairs_id':pairs_id,          
                'neigh_rels_id':neigh_rels_id,
                }

def test_trips_group(args, basedata, saved_model_path_rank, saved_model_path_rerank, omega, outfile):
    from torch.utils.data import DataLoader
    from sentence_transformers.cross_encoder import CrossEncoder
    from main_retrieval_rerank_20k import testmodel_rerank_retrieval
    from main import testmodel
    from retrieve import read_retrieval_data_v1_v2

    if torch.cuda.is_available():
        args.device = 'cuda:' + str(args.cuda)
    else:
        args.device = 'cpu'

    # load model1   "BertResNet_2Inp"
    edges_np = torch.tensor(basedata.edges, dtype=torch.long).to(args.device)     
    node_id = torch.arange(0, basedata.num_nodes, dtype=torch.long).to(args.device)
    model_1 = BertResNet_2Inp(args, edges_np, node_id)
    model_1 = model_1.to(args.device)
    checkpoint = torch.load(saved_model_path_rank)
    model_1.eval()
    model_1.load_state_dict(checkpoint['state_dict'])
    
    # load model2    CrossEncoder
    model_2 = CrossEncoder(saved_model_path_rerank)
    retrieval_datadir = os.path.join(args.results_dir, 'retrieval')   # TODO
    retrieval_datadir = os.path.join(retrieval_datadir, args.dataset) 
    retrieval_datadir = os.path.join(retrieval_datadir, args.retrieval_dirname)  
    retrieval_data_list = read_retrieval_data_v1_v2(retrieval_datadir, args.retrieval_version, args.retrieval_Top_K)

    tailnpfreq2perf_dict = defaultdict(dict)
    test_trips = basedata.test_trips
    
    # model1 test 
    dataset_test = SplittestDataset(args, basedata, test_trips)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, 
                                            num_workers=args.num_workers,collate_fn=dataset_test.get_batch_Local_rps_test)    
    scores_model1_test_set = testmodel(model_1, dataloader_test, args)
    test_perf, test_ranks, test_ranked_cands, rr_list_rank = evaluate_perf_tail(test_trips, scores_model1_test_set, args, basedata, K=10)
    tailnpfreq2perf_dict['rank_perf'] = test_perf

    # model2 test
    dataset_rerank_test = DatasetFromScores_Retrieval_ReRank(args, basedata, scores_model1_test_set, test_trips, data1=retrieval_data_list[2])
    dataloader_rerank_test = DataLoader(dataset=dataset_rerank_test, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers,collate_fn=dataset_rerank_test.get_batch_retrival_data)
    integration_mth_2stage = 1
    args.omega = omega
    test_sorts, test_details = testmodel_rerank_retrieval(model_2, dataloader_rerank_test, args, integration_mth_2stage)  
    test_perf, test_ranks, test_ranked_cands, rr_list_rerank = evaluate_perf_sort_tail(test_trips, test_sorts, args, basedata, K=args.rerank_Top_K)
    tailnpfreq2perf_dict['rerank_perf'] = test_perf

    # 
    rr_dict_rank, rr_dict_rerank = {}, {}
    for i,triple in enumerate(basedata.test_trips):      
        triple_ = ' '.join([str(triple[0]), str(triple[1]), str(triple[2])])
        rr_dict_rank[triple_] = rr_list_rank[i]
        rr_dict_rerank[triple_] = rr_list_rerank[i]

    tailnpfreq2perf_dict['rr_dict_rank'] = rr_dict_rank
    tailnpfreq2perf_dict['rr_dict_rerank'] = rr_dict_rerank
    
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tailnpfreq2perf_dict))
    
    return 

def read_trips_group_perf(filename):
    with open(filename,"r") as f:    #设置文件对象
        tailnpfreq2perf_str = f.read() 
    dict_temp = json.loads(tailnpfreq2perf_str)
    return dict_temp

def plot_trips_group_perf(bound, tailnpfreq2perf, plotfile, tailnpfreq2triple):
    bar_mrr_rank = []
    bar_mrr_rerank = []
    mrr_rank_check = 0 
    mrr_rerank_check = 0 
    num_check = 0
    for i in range(len(bound)):
        mrr_rank = 0
        mrr_rerank = 0
        num = 0
        if i+1<len(bound):
            bound_end = bound[i+1]
        else:
            bound_end = max([k for k in tailnpfreq2triple.keys()])+1
        for freq in range(bound[i], bound_end):
            if freq in tailnpfreq2triple:
                for triple in tailnpfreq2triple[freq]:
                    triple_ = ' '.join([str(triple[0]), str(triple[1]), str(triple[2])])
                    mrr_rank += tailnpfreq2perf['rr_dict_rank'][triple_] 
                    mrr_rerank += tailnpfreq2perf['rr_dict_rerank'][triple_] 
                    num+=1
        if num != 0:
            bar_mrr_rank.append(mrr_rank/num)
            bar_mrr_rerank.append(mrr_rerank/num)
        else:
            bar_mrr_rank.append(0)
            bar_mrr_rerank.append(0)

        mrr_rank_check += mrr_rank
        mrr_rerank_check += mrr_rerank
        num_check += num
    mrr_rank_check = mrr_rank_check/num_check
    mrr_rerank_check = mrr_rerank_check/num_check

    print('total:', tailnpfreq2perf['rank_perf']['mrr'], tailnpfreq2perf['rerank_perf']['mrr'])
    print(num_check, mrr_rank_check, mrr_rerank_check)
    

    
    if len(bound) == 8:
        tick_label = ['[1,2)', '[2,3)', '[3,5)', '[5,10)', '[10,25)', '[25,50)', '[50,100)', '100+']
    else:
        tick_label = ['[0,1)', '[1,2)', '[2,3)', '[3,5)', '[5,10)', '[10,25)', '[25,50)', '[50,100)', '100+']
    # mpl.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(figsize=(5,6))   # 宽x高   (7,3)


    ##### 堆积柱状图
    # x = [i for i in range(len(bound))]
    # plt.bar(x, bar_mrr_rank, align="center", color="orange", label="CEKFA-KFR")
    # y2 = [bar_mrr_rerank[i] - bar_mrr_rank[i] for i in range(len(bar_mrr_rerank))]
    # plt.bar(x, y2, bottom=bar_mrr_rank, align="center", color="blue", label="CEKFA", tick_label=tick_label)
    
    ##### 多组数据的柱状图 
    n_groups = len(bound)
    x = np.arange(n_groups)
    total_width, n = 0.5, 2   # n:每组柱子下的数据组数     wyl
    width = total_width / n
    x = x - (total_width - width) / 2
    
    plt.bar(x, bar_mrr_rank, width = width, label = 'CEKFA-KFARe', fc = 'orange')
    plt.bar(x+width, bar_mrr_rerank, width = width, label = 'CEKFA', fc = 'blue')

    plt.xticks([index + width/2 for index in x], [y[:] for y in tick_label],fontsize=8,fontproperties = 'Times New Roman')  #  todo    tick_label 里面的字符太长了
    plt.yticks(fontsize=8,fontproperties = 'Times New Roman')

    for i in range(len(bound)):
        text = '+'+str(bar_mrr_rerank[i]-bar_mrr_rank[i])[1:6]
        if bar_mrr_rerank[i]-bar_mrr_rank[i] == 0:
            plt.text(i-width/2, bar_mrr_rerank[i]+0.005, '+.00', fontsize=7)
        else:
            plt.text(i-width, bar_mrr_rerank[i]+0.005, text, fontsize=7)

    ##### 
    plt.tick_params(pad = 0.01)
    plt.xlabel("in-degree of tail NP", fontsize=8,fontproperties = 'Times New Roman')
    plt.ylabel("MRR", fontsize=8,fontproperties = 'Times New Roman')
    plt.ylim(0,0.71)
    plt.legend(loc='best',fontsize=8)
    plt.savefig(plotfile)

    return

def main_plot_rp_emb(file_prefix):
    lines = [
        # '15063	was called:	13383	was also named	0.9462819695472717;	16604	was originally called	0.9454656839370728;	4356	was then renamed	0.9424322247505188;	12059	was also a name for	0.9416919350624084;	12672	was once called	0.9409686326980591;	21090	was later re named	0.9409130215644836;	9516	was later renamed to	0.9407490491867065;	6742	was a designation for	0.9402023553848267;	9741	was the real name of	0.9400752186775208;	21500	was later changed to	0.9389989972114563',
        # '16754	was a different name for:	19326	was another name for	0.9703336954116821;	9476	was an alternate name for	0.9610127806663513;	16870	had another name for	0.9553003907203674',
        # '20228	was previously called:	13786	was known previously as	0.9514162540435791;	8974	was earlier called	0.9510113596916199;	19175	was originally named	0.9463474154472351;	17735	was earlier called as	0.9442480206489563;	16094	was previously known as	0.9424027800559998',
        '6731	wrote a commentary on:	13304	published the first edition of	0.9091829657554626;	5489	wrote extensively on	0.9069934487342834;	18795	discusses	0.9043498039245605;	9245	published in	0.9026170372962952;	10595	discusses the efforts of	0.9010642766952515;	3169	has written a book about	0.9000219702720642;	8469	is an author of	0.8995978236198425;	13645	published his treatise on	0.8988016247749329',
        '12072	published an essay on:	7916	wrote an article on	0.9521039724349976;	7945	wrote an essay on	0.9494894742965698;	2073	has published an article on	0.919589638710022;	17303	has written articles published in	0.9189760684967041;	18413	recently published an article on	0.9124444127082825;	3626	has contributed essays to	0.8991003036499023',
        '16369	writes about:	3169	has written a book about	0.9000219702720642;	5384	wrote an article about	0.9340601563453674;	6053	wrote an interesting article about	0.9345722794532776',
        '12064	went away on:	16308	went away for	0.9617084860801697;	14174	is still there on	0.9311728477478027;	1241	vanished from	0.9209498167037964;	19332	is no longer on	0.9192003607749939;	14785	almost disappeared from	0.9178264737129211;	15730	eventually went back to	0.914381206035614;	16367	is gone to	0.9133179783821106;	9675	is gone as is	0.9121794700622559;	16510	went the way of	0.9105837941169739;	16141	was later removed to	0.9098718166351318;	11258	was switched to	0.9073216915130615;	13173	was quickly replaced by	0.9070342183113098;	12279	is no longer for	0.9066478610038757;	13273	took the place of	0.9036542177200317;	16562	has gone back to	0.9028704166412354;	19069	has given way to	0.8994676470756531;	3795	is now off	0.8991498947143555;	20415	gave way to	0.8991382122039795',
        '2295	is useful for:	15882	can be very useful for	0.9731185436248779;	341	is also good for	0.9586003422737122;	12109	is convenient for	0.9582016468048096;	5762	is great for	0.9579318761825562;	3609	is served well by	0.9560786485671997;	1252	is good for	0.9552590250968933;	10489	can be good for	0.9514385461807251;	207	makes great use of	0.9470677971839905;	8141	is also excellent for	0.9469335675239563;	20357	could be usefull for	0.9361482858657837;	19211	was useful in	0.9345079064369202;	15412	is good in	0.9338198900222778;	15617	can also be used for	0.9334093332290649;	5392	also provides	0.9333527684211731;	3991	is utilized for	0.9330556392669678;	7395	is recommended for	0.9328489899635315;	2044	is used to	0.9327766299247742;	21308	is best for	0.9324113130569458;	11041	serves a similar function to	0.9279349446296692;	6293	should make more use of	0.9270142912864685',
        '11275	is central to:	13628	is a central part of	0.9648624658584595;	17122	is central in	0.9633844494819641;	11263	is a central figure in	0.9339505434036255;	11435	is an essential part of	0.9287860989570618;	20489	is involved in	0.927038848400116;	1673	is an important part of	0.9252173900604248;	9898	is closely connected to	0.9237340688705444;	12994	is the centerpiece of	0.9223132133483887;	435	is also involved in	0.9202588796615601;	5204	is the central hub of	0.9174239039421082;	17278	is important in	0.9165710806846619;	19626	lies primarily in	0.9150991439819336;	3915	are also involved in	0.9142951965332031;	18362	is the central figure in	0.9139760136604309;	18301	plays a role in	0.9138244390487671;	14837	is closely linked to	0.9132642149925232;	11074	is intimately involved in	0.9126585125923157;	19116	is the central figure of	0.9121583700180054;	14322	is also an integral part of	0.911758542060852;	19344	has a small part in	0.9115636944770813',
        '19189	separated from:	6866	is separated from	0.9773934483528137;	20376	separates from	0.9565073251724243;	8632	split from	0.9311997890472412;	3748	is split from	0.931147038936615;	13455	is separate from	0.9292404055595398;	7983	had been separated from	0.9193666577339172;	18783	has been separated from	0.917429506778717;	11082	splits from	0.9145405292510986;	15727	is part of	0.9085403084754944;	19933	is one part of	0.9077935814857483;	5553	is another part of	0.9068537950515747;	12008	is a part of	0.903843104839325;	8710	broke off from	0.9020779728889465;	16305	is accompanied by	0.9016632437705994;	6043	appears in	0.901647686958313;	12168	forms one part of	0.9006360769271851',
        '18115	is divided into:	768	is basically split into	0.9061233401298523;	17451	are divided into	0.9054810404777527;	6702	was divided into	0.9025540947914124',
        
        ]

    show_idx_list = [
                    # 16754, 19326, 16870, # another name
                    # 20228, 13786, 8974, 19175, 17735, 16094,    # previously called
                    12072, 7916, 7945,   # publish 
                    # 16369, 5384,          # writes about
                    1241, 14785, 16367, # vanished from
                    # 6293, 207,   # make use
                    15412, 19211, 20357, # good, useful
                    18301, 1673, 11435, # role
                    19189, 3748, 8632,  # split from
                    18115, 768, 17451,   # split into
                    
                    ]
    rp_idx_list, rp_label, rp_str_list = [], [], []
    for i, line in enumerate(lines):
        rp_idx_list_i, rp_str_list_i = process_rp_neighbour_line(line)
        rp_label_i = [i+1 for j in range(len(rp_idx_list_i))]
        rp_idx_list.extend(rp_idx_list_i)
        rp_str_list.append(rp_str_list_i)
        rp_label.append(rp_label_i)


    data_ours = get_emb_data_ours(rp_idx_list) 
    result_ours = main_tsne(data_ours)
    plot_embedding_ours(result_ours, rp_label, rp_idx_list, rp_str_list, show_idx_list, 'ours', file_prefix + '-ours')

    data_care = get_emb_data_care(rp_idx_list)
    result_care = main_tsne(data_care)
    plot_embedding_care(result_care, rp_label, rp_idx_list, rp_str_list, show_idx_list, 'care', file_prefix + '-care')

    data_okgit = get_emb_data_okgit(rp_idx_list)
    result_okgit = main_tsne(data_okgit)
    plot_embedding_okgit(result_okgit, rp_label, rp_idx_list, rp_str_list, show_idx_list, 'okgit', file_prefix + '-okgit')
    
    # data_bertinit = get_emb_data_bertinit(rp_idx_list)
    # result_bertinit = main_tsne(data_bertinit)
    # plot_embedding_okgit(result_bertinit, rp_label, rp_idx_list, rp_str_list, show_idx_list, 'bertinit', 'prism-bertinit-new-15')
    
    # plot_merge(data_care, data_okgit, data_ours, rp_label, rp_idx_list, rp_str_list, show_idx_list, 'all-tsne-prism')
    

    
def main_sparsity_np_rp(basedata):
    ####  NP 与 RP #### 规范化前后的 稀疏性
    np2sparsity, rp2sparsity, _, _ = cal_raw_sparsity(basedata)
    np2sparsity_new, rp2sparsity_new, _, _ = cal_new_sparsity(basedata)
    plot_sparsity(np2sparsity, basedata.args.dataset + '_np-sparsity.pdf')
    plot_sparsity(np2sparsity_new, basedata.args.dataset + '_np-sparsity_new.pdf')
    plot_sparsity(rp2sparsity, basedata.args.dataset + '_rp-sparsity.pdf')
    plot_sparsity(rp2sparsity_new, basedata.args.dataset + '_rp-sparsity_new.pdf')

def main_sparsity_np_rp_relative(basedata):
    ####  NP 与 RP #### 规范化前后的 稀疏度变化倍数
    np2sparsity, rp2sparsity, _, _ = cal_raw_sparsity(basedata)
    np2sparsity_new, rp2sparsity_new, _, _ = cal_new_sparsity(basedata)
    cal_sparsity_relative(np2sparsity, np2sparsity_new, 'NP')
    cal_sparsity_relative(rp2sparsity, rp2sparsity_new, 'RP')


def main_frequency_np_rp_bar(basedata):
    ####  NP 与 RP #### 规范化前后的 出现频率, 频数分布直方图
    _, _, np2freq, rp2freq = cal_raw_sparsity(basedata)
    _, _, np2freq_new, rp2freq_new = cal_new_sparsity(basedata)
    plot_frequency_bar(np2freq, basedata.args.dataset + '_np-frequency.pdf')
    plot_frequency_bar(rp2freq, basedata.args.dataset + '_rp-frequency.pdf')
    plot_frequency_bar(np2freq_new, basedata.args.dataset + '_np-frequency_new.pdf')
    plot_frequency_bar(rp2freq_new, basedata.args.dataset + '_rp-frequency_new.pdf')


def main_frequency_np_rp_line(basedata):
    ####  NP 与 RP #### 规范化前后的 出现频率, 折线图
    _, _, np2freq, rp2freq = cal_raw_sparsity(basedata)
    _, _, np2freq_new, rp2freq_new = cal_new_sparsity(basedata)
    plot_frequency_line(np2freq, np2freq_new, basedata.args.dataset + '_np-frequency-line.pdf', 'NPs')
    plot_frequency_line(rp2freq, rp2freq_new, basedata.args.dataset + '_rp-frequency-line.pdf', 'RPs')

def main_frequency_np_rp_relative(basedata):
    ####  NP 与 RP #### 规范化前后的 出现频率, 频数分布直方图
    _, _, np2freq, rp2freq = cal_raw_sparsity(basedata)
    _, _, np2freq_new, rp2freq_new = cal_new_sparsity(basedata)
    cal_frequency_relative(np2freq, np2freq_new, 'NP')
    cal_frequency_relative(rp2freq, rp2freq_new, 'RP')

    rp_1 = [v for k,v in rp2freq.items() if v > 20]
    rp_2 = [v for k,v in rp2freq_new.items() if v > 20]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(len(rp2freq), len(rp_1), len(rp_1)/len(rp2freq))
    print(len(rp2freq_new), len(rp_2), len(rp_2)/len(rp2freq_new))
    

def main_sparsity_triple_reranking(basedata):
    #### 根据tail NP的入度in-degree统计测试三元组re-ranking后的性能提升(MRR)   柱状图
    
    if basedata.args.dataset == 'ReVerb20K':    
        tailnpfreq2triple, triple2tailnpfreq = split_testset(basedata)
        rank_modeldir = '/opt/data/private/KGuseBert/RPs-wyl-v9-clean/results/models/'
        saved_model_path_rank = os.path.join(rank_modeldir, 'BertResNet_2Inp_Local_Max5_ReVerb20K_bbu_new_ambv2_ENT_13_04_2022_11:23:40.model.pth')
        rerank_modeldir = "/opt/data/private/KGuseBert/RPs-wyl-v9-clean/results/models/new_rerank_ReVerb20K_/opt/data/private/PretrainedBert/"
        saved_model_path_rerank = os.path.join(rerank_modeldir, "distilbert-base-uncased_ambv2_v2_3_10_2022-04-19_16-20-55")
        omega = 0.5
        # test_trips_group(args, basedata, saved_model_path_rank, saved_model_path_rerank, omega, 'test_trips_perf_20k.txt')
        tailnpfreq2perf = read_trips_group_perf('test_trips_perf_20k.txt')
        bound = [0,1,2,3,5,10,25,50,100]
        plot_trips_group_perf(bound, tailnpfreq2perf, 'test_trips_perf_20k_plot_KFARe.pdf', tailnpfreq2triple)
    elif basedata.args.dataset == 'ReVerb45K':    
        tailnpfreq2triple, triple2tailnpfreq = split_testset(basedata)
        rank_modeldir = '/opt/data/private/KGuseBert/RPs-wyl-v9-clean/results/models/'
        saved_model_path_rank = os.path.join(rank_modeldir, 'BertResNet_2Inp_Local_Max10_ReVerb45K_bbu_new_ambv2_ENT_14_04_2022_17:40:04.model.pth')
        rerank_modeldir = "/opt/data/private/KGuseBert/RPs-wyl-v9-clean/results/models/"
        saved_model_path_rerank = os.path.join(rerank_modeldir, "45k_rerank_distilbert-base-uncased_ambv2_v2_3_3_2022-04-19_11-07-41")
        omega = 0.3
        # test_trips_group(args, basedata, saved_model_path_rank, saved_model_path_rerank, omega, 'test_trips_perf_45k.txt')
        tailnpfreq2perf = read_trips_group_perf('test_trips_perf_45k.txt')
        bound = [0,1,2,3,5,10,25,50,100]
        plot_trips_group_perf(bound, tailnpfreq2perf, 'test_trips_perf_45k_plot_KFARe.pdf', tailnpfreq2triple)
  

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

    set_logger(args)
    basedata = load_basedata(args)
    args.num_nodes = basedata.num_nodes
    args.num_rels = basedata.num_rels

    # if torch.cuda.is_available():
    #     args.device = 'cuda:' + str(args.cuda)
    # else:
    #     args.device = 'cpu'
    # args.device = 'cpu'
    
    # # write_emb_data_ours(args, basedata)

    # print_dataset(basedata)

    # 看rerank的结果
    # process_preds('/opt/data/private/KGuseBert/RPs-wyl-v9-clean/results/preds/new_Retrieve3_Rerank_neg3_BertResNet_2Inp_Local_Max10_ReVerb45K_bbu_new_ambv2_ENT_24_04_2022_15:06:40.test_1_0.3.txt')

    #######################################################
    #### 可视化 care,okgit和ours RP embedding 

    # main_plot_rp_emb('111')
    

    ########################################################
    ####  NP 与 RP #### 规范化前后的稀疏性
    # main_sparsity_np_rp(basedata)
    # 稀疏度变化倍数
    # main_sparsity_np_rp_relative(basedata)
    
    ####  NP 与 RP #### 规范化前后的出现频率
    # 频数分布直方图
    # main_frequency_np_rp_bar(basedata)
    # 折线图
    main_frequency_np_rp_line(basedata)
    # 频率变化倍数
    # main_frequency_np_rp_relative(basedata)

    
    ########################################################
    #### 根据tail NP的入度in-degree统计测试三元组re-ranking后的性能提升(MRR)   柱状图
    #### python experiment_analysis.py --dataset ReVerb20K --model_name BertResNet_2Inp --reverse --name smalltest --lm bert --bmn bert-base-uncased --nfeats 768 -lr 0.0001 --rp_neigh_mth Local --maxnum_rpneighs 5 --rpneighs_filename new_sbert_ambv2_ENT_rpneighs --retrieval_Top_K 3  --rerank_training_neg_K 10 --finetune_rerank_model_dir /opt/data/private/PretrainedBert/distilbert-base-uncased
    #### python experiment_analysis.py --dataset ReVerb45K --model_name BertResNet_2Inp --reverse --name smalltest --lm bert --bmn bert-base-uncased --nfeats 768 -lr 0.0001 --rp_neigh_mth Local --maxnum_rpneighs 10 --rpneighs_filename new_sbert_ambv2_ENT_rpneighs --retrieval_Top_K 3  --rerank_training_neg_K 3 --finetune_rerank_model_dir /opt/data/private/WYL/PretrainedBert/distilbert-base-uncased

    # main_sparsity_triple_reranking(basedata)

    
    ########################################################

    



"""
20K

NP 
前:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
1.00        1.00        0.00        0.00        0.00        1.00        3.00        6.00        15.00       6.00        6.00        5.00        7.00        7.00        8.00        15.00       37.00       85.00       1811.00     9051.00     
后:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
6.00        1.00        0.00        0.00        3.00        4.00        12.00       14.00       29.00       6.00        14.00       12.00       9.00        6.00        10.00       27.00       30.00       96.00       1771.00     9015.00     

RP 
前:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
1.00        0.00        0.00        0.00        0.00        1.00        1.00        1.00        7.00        0.00        1.00        3.00        4.00        8.00        12.00       18.00       31.00       145.00      10823.00    2.00        
后:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
11.00       0.00        0.00        0.00        0.00        31.00       8.00        14.00       73.00       6.00        15.00       15.00       38.00       117.00      114.00      132.00      359.00      1051.00     9073.00     1.00    



NP 11065
[0.00,0.90) [0.90,0.95) [0.95,1.00]
27.00       31.00       11007.00    
[0.00,0.90) [0.90,0.95) [0.95,1.00]
69.00       47.00       10949.00 
-----------------------
RP 11058
[0.00,0.90) [0.90,0.95) [0.95,1.00]
11.00       16.00       11031.00    
[0.00,0.90) [0.90,0.95) [0.95,1.00]
137.00      191.00      10730.00  


NP 11065
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
27.00       31.00       145.00      10862.00   
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
69.00       47.00       163.00      10786.00     
-----------------------
RP 11058
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
11.00       16.00       206.00      10825.00    
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
137.00      191.00      1656.00     9074.00  


NP 11065
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
27.00       31.00       145.00      1811.00     9051.00     
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
69.00       47.00       163.00      1771.00     9015.00 
-----------------------
RP 11058
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
11.00       16.00       206.00      10823.00    2.00        
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
137.00      191.00      1656.00     9073.00     1.00      
 


-----------------------
45K

NP
前:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
5.00        1.00        1.00        0.00        9.00        6.00        9.00        17.00       66.00       9.00        31.00       45.00       32.00       58.00       92.00       150.00      382.00      1469.00     24589.00    37.00 
后:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
14.00       13.00       0.00        10.00       18.00       66.00       69.00       52.00       324.00      86.00       134.00      176.00      144.00      269.00      214.00      490.00      858.00      2584.00     21480.00    7.00      


RP
前:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
1.00        0.00        0.00        0.00        0.00        0.00        0.00        0.00        6.00        2.00        1.00        1.00        1.00        6.00        2.00        9.00        24.00       48.00       2960.00     18562.00  
后:
[0.00,0.10) [0.10,0.20) [0.20,0.30) [0.30,0.40) [0.40,0.50) [0.50,0.60) [0.60,0.70) [0.70,0.80) [0.80,0.90) [0.90,0.91) [0.91,0.92) [0.92,0.93) [0.93,0.94) [0.94,0.95) [0.95,0.96) [0.96,0.97) [0.97,0.98) [0.98,0.99) [0.99,1.00) [1.00,1.00)
17.00       0.00        0.00        0.00        0.00        0.00        0.00        0.00        64.00       24.00       20.00       11.00       23.00       52.00       32.00       106.00      224.00      475.00      10935.00    9640.00     


NP 
前:
[0.00,0.90) [0.90,0.95) [0.95,1.00]
114.00      175.00      26719.00    
后:
[0.00,0.90) [0.90,0.95) [0.95,1.00]
566.00      809.00      25633.00  
-----------------------
RP 
前:
[0.00,0.90) [0.90,0.95) [0.95,1.00]
7.00        11.00       21605.00    
后:
[0.00,0.90) [0.90,0.95) [0.95,1.00]
81.00       130.00      21412.00 



NP 27008
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
114.00      175.00      2093.00     24626.00    
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
566.00      809.00      4146.00     21487.00
-----------------------
RP 21623
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
7.00        11.00       83.00       21522.00    
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00]
81.00       130.00      837.00      20575.00 


NP 27008
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
114.00      175.00      2093.00     24589.00    37.00       
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
566.00      809.00      4146.00     21480.00    7.00  

-----------------------
RP 21623
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
7.00        11.00       83.00       2960.00     18562.00    
[0.00,0.90) [0.90,0.95) [0.95,0.99) [0.99,1.00) [1.00,1.00]
81.00       130.00      837.00      10935.00    9640.00  

"""