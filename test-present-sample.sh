### PairRE
# rank
python CEKFA_KGE/main_rank.py --dataset ReVerb45K --model PairRE --do_train --do_valid --do_test -n 256 -b 1024 -d 768 -g 12.0 -a 1.0 -adv -lr 0.0001 -dr --test_batch_size 32 --max_steps 200000 --bert_init --fc_bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 7 --cuda 2
# rerank
python CEKFA_KGE/main_retrieval_rerank.py --dataset ReVerb45K --model PairRE --do_train --do_valid --do_test -n 256 -b 1024 -d 768 -g 12.0 -a 1.0 -adv -lr 0.0001 -dr --test_batch_size 32 --max_steps 200000 --bert_init --fc_bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 7 --init_checkpoint ./results/ReVerb45K/PairRE_1.0_768_12.0_0.0001_npLAN_rpLocal_Num7_bertinit_fc_2023-05-18 --retrieval_Top_K 5 --cuda 2 


### BertResNet_2Inp
# rank
python CEKFA_conv/main_rank.py --dataset ReVerb20K --model_name BertResNet_2Inp --bmn bert-base-uncased --bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 5 --cuda 2 
# rerank
python CEKFA_conv/main_retrieval_rerank.py --dataset ReVerb20K --model_name BertResNet_2Inp --bmn bert-base-uncased --bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 5 --do_rerank --retrieval_Top_K 10 --saved_model_name_rank ./results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-16

