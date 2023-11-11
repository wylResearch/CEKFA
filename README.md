

Codes for paper accepted by IJCAI 2023: A Canonicalization-Enhanced Known Fact-Aware Framework For Open Knowledge Graph Link Prediction

# Installation
python 3.8 + pytorch

1. run `pip install -r requirements.txt`
2. 安装 torch-geometric和相关依赖包
总结：在确定环境的版本后，手动下载 .whl 安装包；需要先安装 torch-scatter， torch-sparse，才能安装 torch-geometric


- 先看官网安装教程 https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels  

It is recommended to manually download the WHL file for installation
1. 确定环境的torch版本：
```
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```
2. 对应的包对应的网址为：
`https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`,
先去网站看该torch版本下有哪些版本的包可以安装，如 torch-1.8.1+cu102 的链接是 https://data.pyg.org/whl/torch-1.8.1+cu102.html
3. 手动下载torch_scatter，torch_sparse，torch_cluster的安装包
4. 安装下载的 .whl 包
```
pip install torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-XXXXXXXXXXX
pip install torch_cluster-XXXXXXXXXX
```
5. 最后：
```
pip install torch-geometric
```


Codes for the paper accepted by IJCAI 2023: "A Canonicalization-Enhanced Known Fact-Aware Framework For Open Knowledge Graph Link Prediction".

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



