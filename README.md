
# CLM-access:A specialized foundation model provides a brand-new solution for high-dimensional single-cell ATAC-seq data

This repository is the official implementation of CLM-access:A specialized foundation model provides a brand-new solution for high-dimensional single-cell ATAC-seq data. 



## Requirements

To install requirements:

```setup
conda create --name CLM-access  -c pyg -c pytorch -c nvidia -c xformers -c conda-forge -c bioconda 'python==3.10' 'pytorch-cuda==12.1' 'pytorch==2.1.2' torchtriton torchvision cudatoolkit xformers nccl py-opencv
conda activate CLM-access
conda install -c conda-forge deepspeed
conda install -c bioconda scvi-tools
conda install pandas numba scipy seaborn pyarrow scikit-learn poetry numpy
conda install -c conda-forge scanpy==1.10.2
pip install dotmap -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lightning -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchscale -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchtext -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sacred -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install apex -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn==2.5.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scglue -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install snapatac2 -i https://pypi.tuna.tsinghua.edu.cn/simple


```


## Data preprocessing
Process the pre-training files and fine-tuning files in the paper.
```bash
python scripts/run_data_preprocessing_pretrain_data.py
python scripts/run_data_preprocessing_finetune_data.py
```
Process the gene expresstion prediction files in the paper.
```bash
python scripts/run_ATAC_RNA_data_preprocessing.py
```
## Pre-training

To train the model(s) in the paper, run this command:
Pre-training script: `scripts/run_pretrain.py`
Parameter configuration: `configs/config.py` （The path needs to be modified when in use）
`ATAC` Pre-training
```train
python run_pretrain.py with pretrain_atac --force
```

## Fine-tuning

To fine-tuning Remove the batch effect and cell type annotation  in the paper, run this command:
Fine-tuning script: `scripts/run_finetune.py`
Parameter configuration: `configs/config_finetune.py` （The path needs to be modified when in use）
`ATAC` Fine-tuning
```train
python run_finetune.py with finetune_atac --force
```
To fine-tuning  gene expresstion prediction in the paper, run this command:
Fine-tuning script: `scripts/run_finetune_RNA.py`
Parameter configuration: `configs/config_finetune.py` （The path needs to be modified when in use）
`ATAC` Fine-tuning
```train
python run_finetune.py with finetune_atac_rna --force
```


## Evaluation

inference scripts: `scripts/run_inference.py`

Parameter configuration: `configs/config_eval.py` （The path needs to be modified when in use）


The function 'infer_atac' is used for cell type label clustering of the 'ATAC' mode.
```bash
python run_inference.py with infer_atac --force
```
The function 'infer_cluster_atac' is used for the clustering of the 'ATAC' mode without cell type labels
```bash
python run_inference_cluster.py with infer_cluster_atac --force
```
The function 'infer_celltype_atac' is used for the evaluation of cell type annotation in downstream tasks of the 'ATAC' mode
```bash
python run_inference_cell_type.py with infer_celltype_atac --force
```
The function 'infer_atac' is used for the evaluation of removing batch effects in downstream tasks of the 'ATAC' mode
```bash
python run_inference_batch.py with infer_batch_atac --force
```




















