import os
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np

from data import *
from parse_args import *



def get_queries(basedata, trips, sep_id_rp):
    """
    数据里的 hr(没有尾实体),但是考虑 反关系带来的影响
    """
    corpus_hr = []
    for trip in trips:
        h,r,t = trip
        if r < sep_id_rp: 
            hr_str =  basedata.id2ent[h]  + ' ' + basedata.id2rel[r] + ' ' + '[ENT].'
        else:
            r_inv = basedata.inverse2rel_map[r]
            hr_str =  '[ENT]' + ' ' + basedata.id2rel[r_inv]  + ' ' + basedata.id2ent[h] + '.'
        corpus_hr.append(hr_str)
    return corpus_hr


def semantic_search(corpus_embeddings, queries, retrieval_Top_K, mode, bi_encoder, idx=None):
    """
    Bi-Encoders (Retrieval) 
    produce embeddings independently for your paragraphs and for your search queries.
    如果输入的 queries 的 mode 是 'train', 而且 这些 queries 的数目和 corpus 是对应的，则不用输入 idx;
    否则若输入的 queries 的 mode 是 'train', 而且 其实只含有一个query,则需要输入其在 corpus 中对应的 idx
    """
    search_results = []
    
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(retrieval_Top_K, corpus_embeddings.shape[0])
    for i, query in enumerate(queries):
        query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
        # print("\n\n======================\n\n")
        # print("Query:", query)
        
        ########## 法一     use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        if mode == "train":
            cos_scores[i] = -1      
        top_results = torch.topk(cos_scores, k=top_k)
        results = [{'corpus_id': idx.item(), 'score': score.item()} for score, idx in zip(top_results[0], top_results[1])]
        search_results.append(results)
        # print("\nTop 5 most similar sentences in corpus:")
        # for score, idx in zip(top_results[0], top_results[1]):
        #     print(corpus[idx], "(Score: {:.4f})".format(score))
        
        ########### 法二    对于query也是训练集(topK中就不能包含自己),这个方法不太方便修改
        # hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
        # hits = hits[0]  # Get the hits for the first query
        # search_results.append(hits)
        # for hit in hits:
        #     print("\t{:.3f}\t{}".format(hit['score'], corpus[hit['corpus_id']].replace("\n", " ")))
    return search_results


def bm25_tokenizer(text):
    # lower case text and remove stop-words from indexing
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


def get_corpus_bm25(corpus):
    tokenized_corpus = []
    for passage in tqdm(corpus):
        tokenized_corpus.append(bm25_tokenizer(passage))
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def BM25_search(corpus_bm25, queries, retrieval_Top_K, mode):
    """ (lexical search) """
    search_results = []
    for i, query in enumerate(queries):
        bm25_scores = corpus_bm25.get_scores(bm25_tokenizer(query))
        if mode == "train":
            bm25_scores[i] = -1       
        top_k = np.argpartition(bm25_scores, -retrieval_Top_K)[-retrieval_Top_K:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_k]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
        search_results.append(bm25_hits)
        
        # print("Top-3 lexical search (BM25) hits")
        # for hit in bm25_hits:
        #     print("\t{:.3f}\t{}".format(hit['score'], corpus[hit['corpus_id']].replace("\n", " ")))
    return search_results


def get_canonical_triples_sbert(args, basedata):
    retrieval_Top_K = args.retrieval_Top_K                          # Number of passages we want to retrieve
    retrieval_version = args.retrieval_version
    sep_id_rp = len(basedata.rel2id) / 2 if args.reverse else len(basedata.rel2id)
    
    bi_encoder = SentenceTransformer(args.retrieval_model)
    bi_encoder.max_seq_length = 384     #Truncate long passages to 256 tokens
    
    corpus_trips = basedata.train_trips
    corpus = get_queries(basedata, corpus_trips, sep_id_rp)
    corpus_embeddings = bi_encoder.encode(corpus, show_progress_bar=True, convert_to_tensor=True)
    for mode in ['train', 'test', 'valid']:
        if mode == 'train':
            queries_trips = basedata.train_trips
        elif mode == 'valid':
            queries_trips = basedata.valid_trips
        elif mode == 'test':
            queries_trips = basedata.test_trips
        
        queries = get_queries(basedata, queries_trips, sep_id_rp)
        
        file_name = "retrieval_" + retrieval_version + "_" + mode + '_Top' + str(retrieval_Top_K)

        ##### Sematic Search #####
        dir_name = args.retrieval_dirname
        search_results_ss = semantic_search(corpus_embeddings, queries, retrieval_Top_K, mode, bi_encoder)
                                            
        print("semantic_search over.")
        write_canonical_triples(args, basedata, args.dataset, corpus_trips, queries_trips, corpus, queries, search_results_ss, dir_name, file_name, retrieval_version, sep_id_rp)
        print("write semantic_search_results over.")
    return


def get_canonical_triples_BM25(args, basedata):
    retrieval_Top_K = args.retrieval_Top_K                          # Number of passages we want to retrieve
    retrieval_version = args.retrieval_version
    sep_id_rp = len(basedata.rel2id) / 2 if args.reverse else len(basedata.rel2id)
    
    corpus_trips = basedata.train_trips
    corpus = get_queries(basedata, corpus_trips, sep_id_rp)
    corpus_bm25 = get_corpus_bm25(corpus)
    for mode in ['train', 'test', 'valid']:
        if mode == 'train':
            queries_trips = basedata.train_trips
        elif mode == 'valid':
            queries_trips = basedata.valid_trips
        elif mode == 'test':
            queries_trips = basedata.test_trips
        
        queries = get_queries(basedata, queries_trips, sep_id_rp)
        
        file_name = "retrieval_" + retrieval_version + "_" + mode + '_Top' + str(retrieval_Top_K)

        print("start BM25_search")
        dir_name = args.retrieval_dirname
        search_results_bm = BM25_search(corpus_bm25, queries, retrieval_Top_K, mode)
        print("BM25_search over.")
        write_canonical_triples(args, basedata, args.dataset, corpus_trips, queries_trips, corpus, queries, search_results_bm, dir_name, file_name, retrieval_version, sep_id_rp)
        print("write BM25_search_results over.")
    return
    

def write_canonical_triples(args, basedata, dataset, corpus_trips, queries_trips, corpus, queries, search_results, dir_name, file_name, version, sep_id_rp):
    """ 将v1/v2版本的搜索结果写到文件里"""
    retrieval_datadir = os.path.join(args.retrieval_basedir, args.retrieval_dirname)  
    if not os.path.exists(retrieval_datadir):
        os.makedirs(retrieval_datadir)

    with open(os.path.join(retrieval_datadir, file_name + '.txt'), 'w') as fw:
        with open(os.path.join(retrieval_datadir, file_name + '_str.txt'), 'w') as fw2:
            for i in range(len(search_results)):
                query_trip = queries_trips[i]
                query = queries[i]
                search_result = search_results[i]
                corpus_idx = [s['corpus_id'] for s in search_result]
                trips = [corpus_trips[idx] for idx in corpus_idx]
                if version == 'v1':
                    search_string = '; '.join([basedata.id2ent[trip[0]] + ' ' + basedata.id2rel[trip[1]] + ' ' + basedata.id2ent[trip[2]]  for trip in trips])
                elif version in ['v2', 'v3']:
                    string_list = []
                    for trip in trips:
                        h,r,t = trip
                        if r < sep_id_rp: 
                            hrt_str =  basedata.id2ent[h]  + ' ' + basedata.id2rel[r] + ' ' + basedata.id2ent[t]
                        else:
                            r_inv = basedata.inverse2rel_map[r]
                            hrt_str =  basedata.id2ent[t] + ' ' + basedata.id2rel[r_inv]  + ' ' + basedata.id2ent[h]
                        string_list.append(hrt_str)
                    search_string = '; '.join(string_list)
                if version in ['v1', 'v2']:
                    fw.write("%s\t%s\t%s\n" % (query_trip[0], query_trip[1], search_string))
                    fw2.write("%s\t%s\t%s\n" % (basedata.id2ent[int(query_trip[0])], basedata.id2rel[int(query_trip[1])], search_string))
                elif version =='v3':
                    fw.write("%s\t%s\t%s\t%s\n" % (query_trip[0], query_trip[1], query_trip[2], search_string))
                    fw2.write("%s\t%s\t%s\t%s\n" % (basedata.id2ent[int(query_trip[0])], basedata.id2rel[int(query_trip[1])], basedata.id2ent[int(query_trip[2])], search_string))


def read_canonical_triples(retrieval_datadir, version, retrieval_Top_K):
    retrieval_data_list = [[], [], []]
    for i, mode in enumerate(['train', 'valid', 'test']):
        file_name = "retrieval_" + version + "_" + mode + '_Top' + str(retrieval_Top_K) +'.txt'
        for line in open(os.path.join(retrieval_datadir, file_name), 'r'):
            line = line.strip().split('\t')
            h, r, retrieval_data = int(line[0]), int(line[1]), line[2]
            retrieval_data_list[i].append([h, r, retrieval_data])
    return retrieval_data_list
