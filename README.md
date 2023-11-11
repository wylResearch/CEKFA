
Codes for the paper accepted by IJCAI 2023: "A Canonicalization-Enhanced Known Fact-Aware Framework For Open Knowledge Graph Link Prediction".
s
### Environment Configuration

The project requires Python 3.8 and PyTorch. Please follow the steps below to set up the environment:

1. Run the following commands:
    ```
    conda create --name cekfa python=3.8
    conda activate cekfa
    pip install -r requirements.txt
    ```

2. Install torch-geometric and its dependencies (Please refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels)). It is recommended to manually download the corresponding .whl installation packages.

   - Determine the torch version of your environment by running the following commands:
     ```
     python -c "import torch; print(torch.__version__)"
     python -c "import torch; print(torch.version.cuda)"
     ```

   - The URL for the corresponding packages is `https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`. For example, if you have torch-1.8.1+cu102, the URL would be [https://data.pyg.org/whl/torch-1.8.1+cu102.html](https://data.pyg.org/whl/torch-1.8.1+cu102.html).

   - From the obtained URL，manually download the packages for torch_scatter, torch_sparse, and torch_cluster.

   - Install the downloaded .whl packages using the following commands:
     ```
     pip install torch_scatter-XXXXXXXXXXX.whl
     pip install torch_sparse-XXXXXXXXXXX.whl
     pip install torch_cluster-XXXXXXXXXX.whl
     ```

    - Finally, install torch-geometric by running the command:
        ```
        pip install torch-geometric
        ```
3. Install pre-trained language models: 
    - BERT: [bert-base-uncased](https://huggingface.co/bert-base-uncased)
    - SBERT: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)


### Running Commands
Example commands are listed in test-present-sample.sh:
- PairRE

    - rank
    ```
    python CEKFA_KGE/main_rank.py --dataset ReVerb45K --model PairRE --do_train --do_valid --do_test -n 256 -b 1024 -d 768 -g 12.0 -a 1.0 -adv -lr 0.0001 -dr --test_batch_size 32 --max_steps 200000 --bert_init --fc_bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 7 --cuda 2
    ```

    - rerank
    ```
    python CEKFA_KGE/main_retrieval_rerank.py --dataset ReVerb45K --model PairRE --do_train --do_valid --do_test -n 256 -b 1024 -d 768 -g 12.0 -a 1.0 -adv -lr 0.0001 -dr --test_batch_size 32 --max_steps 200000 --bert_init --fc_bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 7 --init_checkpoint ./results/ReVerb45K/PairRE_1.0_768_12.0_0.0001_npLAN_rpLocal_Num7_bertinit_fc_2023-05-18 --retrieval_Top_K 5 --cuda 2 
    ```

- KG-ResNet
    - rank
    ```
    python CEKFA_conv/main_rank.py --dataset ReVerb20K --model_name BertResNet_2Inp --bmn bert-base-uncased --bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 5 --cuda 2 
    ```
    - rerank
    ```
    python CEKFA_conv/main_retrieval_rerank.py --dataset ReVerb20K --model_name BertResNet_2Inp --bmn bert-base-uncased --bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 5 --do_rerank --retrieval_Top_K 10 --saved_model_name_rank ./results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-16
    ```



