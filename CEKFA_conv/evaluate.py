import numpy as np
import matplotlib.pyplot as plt


def get_rank(scores, clust, Hits, entid2clustid, filter_clustID, candidates, K=10, score_flag=True):
    """
    score_flag 为True 表示输入的scores是分数(还没排序); False 表示输入scores 已经为排序后的sort
    """
    hits = np.ones((len(Hits)))
    if score_flag:
        scores = np.argsort(scores)     # 将输入中的元素从小到大排序后，提取对应的索引index，而该index其实也可以看作排序后的实体id
    rank = 1
    high_rank_clust = set()
    for i in range(scores.shape[0]):        # num_nodes
        if scores[i] not in candidates:
            continue
        if scores[i] in clust:              # clust ：正确的目标实体的id集合
            break
        else:           # 属于没有出现过的cluster(新的)，且不在需要排除的cluster中，才会对rank计数
            if entid2clustid[scores[i]] not in high_rank_clust and entid2clustid[scores[i]] not in filter_clustID:
                rank+=1
                high_rank_clust.add(entid2clustid[scores[i]])
    for i,r in enumerate(Hits):
        if rank>r:
            hits[i]=0
        else:
            break
    count = 0
    top_cands = np.zeros((K,), dtype=np.int32)
    for score in scores:
        if score not in candidates:
            continue
        else:
            top_cands[count] = score
            count += 1
        if count >= K:
            break
    return rank, hits, top_cands



def evaluate_perf(triples, scores, args, basedata, K=10):
    # triples:array  (num_samples,3)
    # scores:array   (num_samples, num_nodes)
    head = triples[:,0]
    rel = triples[:,1]
    tail = triples[:,2]

    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(args.Hits)))
    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    true_clusts = basedata.true_clusts
    entid2clustid = basedata.entid2clustid
    ent_filter = basedata.label_filter

    ranked_cands = np.zeros((triples.shape[0], K), dtype=np.int32)
    candidates = set(basedata.id2ent.keys())
    
    for j in range(triples.shape[0]):
        sample_scores = -scores[j,:]
        t_clust = set(true_clusts[tail[j]])             # 正确的目标实体的id集合
        _filter = []
        if (head[j],rel[j]) in ent_filter:              # h,r 出现在训练数据中
            _filter = ent_filter[(head[j],rel[j])]      # 需要filter的cluster ids
        if j%2==1:
            H_r, H_h, cur_ranked_cands = get_rank(sample_scores, t_clust, args.Hits, entid2clustid, _filter, candidates, K)
            H_Rank.append(H_r)
            H_inv_Rank.append(1/H_r)
            H_Hits += H_h
        else:   # h,r,(t)
            T_r, T_h, cur_ranked_cands = get_rank(sample_scores, t_clust, args.Hits, entid2clustid, _filter, candidates, K)
            T_Rank.append(T_r)
            T_inv_Rank.append(1/T_r)
            T_Hits += T_h
        ranked_cands[j,:] = cur_ranked_cands
    mean_rank_head = np.mean(np.array(H_Rank))
    mean_rank_tail = np.mean(np.array(T_Rank))
    mean_rank = 0.5*(mean_rank_head+mean_rank_tail) 
    mean_inv_rank_head = np.mean(np.array(H_inv_Rank))
    mean_inv_rank_tail = np.mean(np.array(T_inv_Rank))
    mean_inv_rank = 0.5*(mean_inv_rank_head+mean_inv_rank_tail)
    hits_at_head = {}
    hits_at_tail = {}
    hits_at = {}
    for i, hits in enumerate(args.Hits):        # 默认：[1,3,10,30,50]
        hits_at_head[hits] = H_Hits[i]/len(H_Rank)
        hits_at_tail[hits] = T_Hits[i]/len(T_Rank)
        hits_at[hits] = 0.5*(hits_at_head[hits]+hits_at_tail[hits])
    perf = {'mr': mean_rank,
            'mrr': mean_inv_rank,
            'hits@': hits_at,
            'head_mr': mean_rank_head,
            'head_mrr': mean_inv_rank_head,
            'head_hits@': hits_at_head,
            'tail_mr': mean_rank_tail,
            'tail_mrr': mean_inv_rank_tail,
            'tail_hits@': hits_at_tail,
            }
    # print(H_Rank[:100])
    return perf, {'tail':T_Rank, 'head':H_Rank}, ranked_cands



def evaluate_perf_sort(triples, sorts, args, basedata, K=10):
    """
    输入是 sorts 不是 scores
    """
    # triples:array  (num_samples,3)
    # scores:array   (num_samples, num_nodes)
    head = triples[:,0]
    rel = triples[:,1]
    tail = triples[:,2]

    H_Rank = []
    H_inv_Rank = []
    H_Hits = np.zeros((len(args.Hits)))
    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    true_clusts = basedata.true_clusts
    entid2clustid = basedata.entid2clustid
    ent_filter = basedata.label_filter

    ranked_cands = np.zeros((triples.shape[0], K), dtype=np.int32)
    candidates = set(basedata.id2ent.keys())
    
    for j in range(triples.shape[0]):
        sample_sorts = sorts[j,:]
        t_clust = set(true_clusts[tail[j]])             # 正确的目标实体的id集合
        _filter = []
        if (head[j],rel[j]) in ent_filter:              # h,r 出现在训练数据中
            _filter = ent_filter[(head[j],rel[j])]      # 需要filter的cluster ids
        if j%2==1:
            H_r, H_h, cur_ranked_cands = get_rank(sample_sorts, t_clust, args.Hits, entid2clustid, _filter, candidates, K, score_flag=False)
            H_Rank.append(H_r)
            H_inv_Rank.append(1/H_r)
            H_Hits += H_h
        else:   # h,r,(t)
            T_r, T_h, cur_ranked_cands = get_rank(sample_sorts, t_clust, args.Hits, entid2clustid, _filter, candidates, K, score_flag=False)
            T_Rank.append(T_r)
            T_inv_Rank.append(1/T_r)
            T_Hits += T_h
        ranked_cands[j,:] = cur_ranked_cands
    mean_rank_head = np.mean(np.array(H_Rank))
    mean_rank_tail = np.mean(np.array(T_Rank))
    mean_rank = 0.5*(mean_rank_head+mean_rank_tail) 
    mean_inv_rank_head = np.mean(np.array(H_inv_Rank))
    mean_inv_rank_tail = np.mean(np.array(T_inv_Rank))
    mean_inv_rank = 0.5*(mean_inv_rank_head+mean_inv_rank_tail)
    hits_at_head = {}
    hits_at_tail = {}
    hits_at = {}
    for i, hits in enumerate(args.Hits):        # 默认：[1,3,10,30,50]
        hits_at_head[hits] = H_Hits[i]/len(H_Rank)
        hits_at_tail[hits] = T_Hits[i]/len(T_Rank)
        hits_at[hits] = 0.5*(hits_at_head[hits]+hits_at_tail[hits])
    perf = {'mr': mean_rank,
            'mrr': mean_inv_rank,
            'hits@': hits_at,
            'head_mr': mean_rank_head,
            'head_mrr': mean_inv_rank_head,
            'head_hits@': hits_at_head,
            'tail_mr': mean_rank_tail,
            'tail_mrr': mean_inv_rank_tail,
            'tail_hits@': hits_at_tail,
            }
    # print(H_Rank[:100])
    return perf, {'tail':T_Rank, 'head':H_Rank}, ranked_cands




def evaluate_perf_tail(triples, scores, args, basedata, K=10):
    # triples:array  (num_samples,3)
    # scores:array   (num_samples, num_nodes)
    head = triples[:,0]
    rel = triples[:,1]
    tail = triples[:,2]

    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    true_clusts = basedata.true_clusts
    entid2clustid = basedata.entid2clustid
    ent_filter = basedata.label_filter

    ranked_cands = np.zeros((triples.shape[0], K), dtype=np.int32)
    candidates = set(basedata.id2ent.keys())
    
    for j in range(triples.shape[0]):
        sample_scores = -scores[j,:]
        t_clust = set(true_clusts[tail[j]])             # 正确的目标实体的id集合
        _filter = []
        if (head[j],rel[j]) in ent_filter:              # h,r 出现在训练数据中
            _filter = ent_filter[(head[j],rel[j])]      # 需要filter的cluster ids

       # h,r,(t)
        T_r, T_h, cur_ranked_cands = get_rank(sample_scores, t_clust, args.Hits, entid2clustid, _filter, candidates, K)
        T_Rank.append(T_r)
        T_inv_Rank.append(1/T_r)
        T_Hits += T_h
        ranked_cands[j,:] = cur_ranked_cands
    mean_rank_tail = np.mean(np.array(T_Rank))
    mean_inv_rank_tail = np.mean(np.array(T_inv_Rank))
    sum_rr = sum(T_inv_Rank)
    hits_at_tail = {}
    for i, hits in enumerate(args.Hits):        # 默认：[1,3,10,30,50]
        hits_at_tail[hits] = T_Hits[i]/len(T_Rank)
    perf = {'mr': mean_rank_tail,
            'mrr': mean_inv_rank_tail,
            'hits@': hits_at_tail,
            'sum_rr':sum_rr,
            }
    # print(H_Rank[:100])
    return perf, {}, ranked_cands, T_inv_Rank



def evaluate_perf_sort_tail(triples, sorts, args, basedata, K=10):
    """
    输入是 sorts 不是 scores
    """
    # triples:array  (num_samples,3)
    # scores:array   (num_samples, num_nodes)
    head = triples[:,0]
    rel = triples[:,1]
    tail = triples[:,2]

    T_Rank = []
    T_inv_Rank = []
    T_Hits = np.zeros((len(args.Hits)))
    true_clusts = basedata.true_clusts
    entid2clustid = basedata.entid2clustid
    ent_filter = basedata.label_filter

    ranked_cands = np.zeros((triples.shape[0], K), dtype=np.int32)
    candidates = set(basedata.id2ent.keys())
    
    for j in range(triples.shape[0]):
        sample_sorts = sorts[j,:]
        t_clust = set(true_clusts[tail[j]])             # 正确的目标实体的id集合
        _filter = []
        if (head[j],rel[j]) in ent_filter:              # h,r 出现在训练数据中
            _filter = ent_filter[(head[j],rel[j])]      # 需要filter的cluster ids
        # h,r,(t)
        T_r, T_h, cur_ranked_cands = get_rank(sample_sorts, t_clust, args.Hits, entid2clustid, _filter, candidates, K, score_flag=False)
        T_Rank.append(T_r)
        T_inv_Rank.append(1/T_r)
        T_Hits += T_h
        ranked_cands[j,:] = cur_ranked_cands
    mean_rank_tail = np.mean(np.array(T_Rank))
    mean_inv_rank_tail = np.mean(np.array(T_inv_Rank))
    sum_rr = sum(T_inv_Rank)
    hits_at_tail = {}
    for i, hits in enumerate(args.Hits):        # 默认：[1,3,10,30,50]
        hits_at_tail[hits] = T_Hits[i]/len(T_Rank)
    perf = {'mr': mean_rank_tail,
            'mrr': mean_inv_rank_tail,
            'hits@': hits_at_tail,
            'sum_rr':sum_rr,
            }
    # print(H_Rank[:100])
    return perf, {}, ranked_cands, T_inv_Rank



def save_ranks(test_triples, ranks, rankfile):
    num_ranks = len(ranks['head']) + len(ranks['tail'])
    tr = np.zeros((num_ranks, 4), dtype=np.int32)
    # tr = np.zeros((len(ranks), 4), dtype=np.int32)
    tr[:,:3] = test_triples
    head_ranks = np.array(ranks['head'])
    tail_ranks = np.array(ranks['tail'])
    all_ranks = np.vstack([tail_ranks, head_ranks]).T.reshape((num_ranks,))
    tr[:,3] = all_ranks
    np.save(rankfile, tr)


def save_preds(test_triples, ranked_cands, data, predsnpy, predstxt):
    preds = np.zeros((test_triples.shape[0], test_triples.shape[1]+ranked_cands.shape[1]), dtype=np.int32)
    preds[:,:3] = test_triples
    preds[:,3:] = ranked_cands
    # np.save(predsnpy, preds)

    delim = "\t"
    K = ranked_cands.shape[1]
    with open(predstxt, 'w') as fout:
        for i in range(test_triples.shape[0]):
            head = data.id2ent[test_triples[i,0]]
            rel = data.id2rel[test_triples[i,1]]
            tail = data.id2ent[test_triples[i,2]]
            topk = [",".join([head, rel, tail])]
            flag = "    --InTopK" if test_triples[i,2] in ranked_cands[i].tolist() else "    --NotInTopK"
            print(topk,flag, file=fout)
            cands = []
            for j in range(K):
                cands.append(data.id2ent[ranked_cands[i,j]])
            print(";\t".join(cands), file=fout)


def save_preds_rerank_details(test_triples, ranked_cands, data, details, predstxt):
    preds = np.zeros((test_triples.shape[0], test_triples.shape[1]+ranked_cands.shape[1]), dtype=np.int32)
    preds[:,:3] = test_triples
    preds[:,3:] = ranked_cands

    test_sorts_raw_K, test_scores_raw_K, test_scores_K, test_scores_raw1_K, test_scores1_K, test_scores_final_K, test_scores_final_K_rerank = details 

    delim = "\t"
    K = min(ranked_cands.shape[1], test_sorts_raw_K.shape[1])
    with open(predstxt, 'w') as fout:
        for i in range(test_triples.shape[0]):
            head = data.id2ent[test_triples[i,0]]
            rel = data.id2rel[test_triples[i,1]]
            tail = data.id2ent[test_triples[i,2]]
            topk = [",".join([head, rel, tail])]
            flag = "    --InTopK" if test_triples[i,2] in test_sorts_raw_K[i].tolist() else "    --NotInTopK"
            print(topk,flag, file=fout)
            cands_raw = []
            cands_scores_raw = []
            cands_scores_raw1 = []
            cands_scores_new = []
            cands_scores_new1 = []
            cands_scores_final = []
            cands_rerank = []
            cands_scores_rerank = []
            for j in range(K):
                cands_raw.append('%-14s'%(data.id2ent[test_sorts_raw_K[i,j]]+";"))
                cands_scores_raw.append("{score:0<6.5f}".format(score = test_scores_raw_K[i,j]))
                cands_scores_raw1.append("{score:0<6.5f}".format(score = test_scores_raw1_K[i,j]))
                cands_scores_new.append("{score:0<6.5f}".format(score = test_scores_K[i,j]))
                cands_scores_new1.append("{score:0<6.5f}".format(score = test_scores1_K[i,j]))
                cands_scores_final.append("{score:0<6.5f}".format(score = test_scores_final_K[i,j]))
                cands_rerank.append('%-14s'%(data.id2ent[ranked_cands[i,j]]+";"))
                cands_scores_rerank.append("{score:0<6.5f}".format(score = test_scores_final_K_rerank[i,j]))
            print("\t".join(cands_raw), file=fout)
            print(";\t\t".join(cands_scores_raw), file=fout)
            print(";\t\t".join(cands_scores_raw1), file=fout)
            print(";\t\t".join(cands_scores_new), file=fout)
            print(";\t\t".join(cands_scores_new1), file=fout)
            print(";\t\t".join(cands_scores_final), file=fout)
            print("\t".join(cands_rerank), file=fout)
            print(";\t\t".join(cands_scores_rerank), file=fout)



def plot_perf(loss_list, mrr_list, mr_list, hit10_list, hit30_list, hit50_list, plot_file):
    plt.subplot(231)
    plt.cla()
    plt.plot(list(range(len(loss_list)))[2:], loss_list[2:], linestyle="-", label="loss")
    plt.legend(loc='upper right')

    plt.subplot(232)
    plt.cla()
    plt.plot([(i+1)*5 for i in range(len(mrr_list))], mrr_list, linestyle="-", label="best valid MRR")
    plt.legend(loc='lower right')

    plt.subplot(233)
    plt.cla()
    plt.plot([(i+1)*5 for i in range(len(mr_list))], mr_list, linestyle="-", label="best valid MR")
    plt.legend(loc='upper right')

    plt.subplot(234)
    plt.cla()
    plt.xlabel("epoch")
    plt.plot([(i+1)*5 for i in range(len(hit10_list))], hit10_list, linestyle="-", label="hits10")
    plt.legend(loc='lower right')

    plt.subplot(235)
    plt.cla()
    plt.xlabel("epoch")
    plt.plot([(i+1)*5 for i in range(len(hit30_list))], hit30_list, linestyle="-", label="hits30")
    plt.legend(loc='lower right')

    plt.subplot(236)
    plt.cla()
    plt.xlabel("epoch")
    plt.plot([(i+1)*5 for i in range(len(hit50_list))], hit50_list, linestyle="-", label="hits50")
    plt.legend(loc='lower right')

    plt.savefig(plot_file, dpi=1000)  


def plot_perf2(logfilename):
    loss_list = []
    mrr_list = []
    mr_list = []
    hit10_list, hit30_list, hit50_list = [], [], []
    for line in open(logfilename, "r"):
        if "epoch_loss: " in line:
            line = line.split("epoch_loss: ")
            loss = float(line[-1])
            epoch = int(line[0].split("epoch ")[-1].split("/")[0])
            loss_list.append(loss)
        elif "Best Valid MRR: " in line:
            line = line.strip().split(": ")
            mrr = float(line[1].split(", Best Valid MR")[0])
            mr = float(line[2].split(", Best Epoch")[0])
            mrr_list.append(mrr)
            mr_list.append(mr)
        elif "hits@1" in line:
            line = line.split(":")
            hit10 = float(line[9].split(",  hits@")[0])
            hit30 = float(line[10].split(",  hits@")[0])
            hit50 = float(line[11].split(",  hits@")[0])
            hit10_list.append(hit10)
            hit30_list.append(hit30)
            hit50_list.append(hit50)



    # plot
    plotfile = logfilename.split(":")[0] + "_loss.png"
    if len(loss_list) > 10:
        plt.subplot(231)
        plt.cla()
        plt.ylabel("loss")
        plt.plot(list(range(len(loss_list)))[2:], loss_list[2:], linestyle="--", label="loss")
        plt.legend(loc='upper right')

        plt.subplot(232)
        plt.cla()
        plt.plot([(i+1)*5 for i in range(len(mrr_list))], mrr_list, linestyle="-", label="best valid MRR")
        plt.legend(loc='lower right')

        plt.subplot(233)
        plt.cla()
        plt.plot([(i+1)*5 for i in range(len(mr_list))], mr_list, linestyle="-", label="best valid MR")
        plt.legend(loc='upper right')

        plt.subplot(234)
        plt.cla()
        plt.xlabel("epoch")
        plt.plot([(i+1)*5 for i in range(len(hit10_list))], hit10_list, linestyle="-", label="hits10")
        plt.legend(loc='lower right')

        plt.subplot(235)
        plt.cla()
        plt.xlabel("epoch")
        plt.plot([(i+1)*5 for i in range(len(hit30_list))], hit30_list, linestyle="-", label="hits30")
        plt.legend(loc='lower right')

        plt.subplot(236)
        plt.cla()
        plt.xlabel("epoch")
        plt.plot([(i+1)*5 for i in range(len(hit50_list))], hit50_list, linestyle="-", label="hits50")
        plt.legend(loc='lower right')


        plt.savefig(plotfile) 

if __name__ == '__main__':
    logfilename3 = "ConvE_ReVerb20K_bbc_07_11_2021_13:52:08.log"    
    plot_perf2(logfilename3)
