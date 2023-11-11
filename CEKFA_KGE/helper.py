import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
import json
import logging

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
     

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


def evaluate_perf_kge(triples, scores, args, basedata, K=10):
    """
    分 head-batch 和 tail-batch
    """
    head_scores, tail_scores = scores

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
    ent_filter_tr2h = basedata.label_filter_tr2h

    ranked_cands = np.zeros((triples.shape[0]*2, K), dtype=np.int32)
    candidates = set(basedata.id2ent.keys())
    
    for j in range(triples.shape[0]):
        ####   tail-batch   # h,r,(t)
        sample_scores_tail = -tail_scores[j,:]          # TODO  按照什么排序
        t_clust_tail = set(true_clusts[tail[j]])             # 正确的目标实体的id集合
        _filter_tail = []
        if (head[j],rel[j]) in ent_filter:              # h,r 出现在训练数据中
            _filter_tail = ent_filter[(head[j],rel[j])]      # 需要filter的cluster ids
        T_r, T_h, cur_ranked_cands_tail = get_rank(sample_scores_tail, t_clust_tail, args.Hits, entid2clustid, _filter_tail, candidates, K)
        T_Rank.append(T_r)
        T_inv_Rank.append(1/T_r)
        T_Hits += T_h
        ranked_cands[j*2,:] = cur_ranked_cands_tail
        
        #### head-batch   # (h),r,t
        sample_scores_head = -head_scores[j,:]
        t_clust_head = set(true_clusts[head[j]])             # 正确的目标实体的id集合
        _filter_head = []
        if (tail[j], rel[j]) in ent_filter_tr2h:              # h,r 出现在训练数据中
            _filter_head = ent_filter_tr2h[((tail[j], rel[j]))]
        H_r, H_h, cur_ranked_cands_head = get_rank(sample_scores_head, t_clust_head, args.Hits, entid2clustid, _filter_head, candidates, K)
        H_Rank.append(H_r)
        H_inv_Rank.append(1/H_r)
        H_Hits += H_h
        ranked_cands[j*2+1,:] = cur_ranked_cands_head

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


def evaluate_perf_kge_v2(triples, scores, args, basedata, K=10):
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
    ent_filter_tr2h = basedata.label_filter_tr2h

    ranked_cands = np.zeros((triples.shape[0], K), dtype=np.int32)
    candidates = set(basedata.id2ent.keys())
    
    for j in range(triples.shape[0]):
        sample_scores = -scores[j,:]
        t_clust = set(true_clusts[tail[j]])                 # 正确的目标实体的id集合
        if j%2==1:           
            _filter = []
            if (head[j], rel[j]) in ent_filter_tr2h:              # h,r 出现在训练数据中
                _filter = ent_filter_tr2h[((head[j], rel[j]))]
            H_r, H_h, cur_ranked_cands = get_rank(sample_scores, t_clust, args.Hits, entid2clustid, _filter, candidates, K)
            H_Rank.append(H_r)
            H_inv_Rank.append(1/H_r)
            H_Hits += H_h
        else:   # h,r,(t)
            _filter = []
            if (head[j],rel[j]) in ent_filter:              # h,r 出现在训练数据中
                _filter = ent_filter[(head[j],rel[j])]      # 需要filter的cluster ids
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


def evaluate_perf_sort_KGE(triples, sorts, args, basedata, K=10):
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
    ent_filter_tr2h = basedata.label_filter_tr2h

    ranked_cands = np.zeros((triples.shape[0], K), dtype=np.int32)
    candidates = set(basedata.id2ent.keys())
    
    for j in range(triples.shape[0]):
        sample_sorts = sorts[j,:]
        t_clust = set(true_clusts[tail[j]])             # 正确的目标实体的id集合
        
        if j%2==1:
            _filter = []
            if (head[j], rel[j]) in ent_filter_tr2h:              # h,r 出现在训练数据中
                _filter = ent_filter_tr2h[((head[j], rel[j]))]
            H_r, H_h, cur_ranked_cands = get_rank(sample_sorts, t_clust, args.Hits, entid2clustid, _filter, candidates, K, score_flag=False)
            H_Rank.append(H_r)
            H_inv_Rank.append(1/H_r)
            H_Hits += H_h
        else:   # h,r,(t)
            _filter = []
            if (head[j],rel[j]) in ent_filter:              # h,r 出现在训练数据中
                _filter = ent_filter[(head[j],rel[j])]      # 需要filter的cluster ids
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


