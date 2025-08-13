import os
from sacred import Experiment

ex = Experiment("Beit3")

CODE_REPO = os.path.dirname(os.path.dirname(__file__))


@ex.config
def config():
    project_name = "CLM-access"
    seed = 1
    data_path = None

    ################################################################################
    # vocab setting
    # 1. dataset, 2. vocab_size, 3. vocab_file
    ################################################################################
    atac_vocab_size = None
    atac_vocab_file = os.path.join(CODE_REPO, "dataset", "ATAC.vocab.json")

    ################################################################################
    # Transformer Setting
    ################################################################################
    encoder_layers = 8 # for debug
    encoder_embed_dim = 256

    encoder_attention_heads = 8
    encoder_ffn_embed_dim =256

    # feedforward activation, relu, gelu, swish
    activation_fn = "gelu"
    activation_dropout = 0.0

    # architecture setting
    pre_norm = False
    multi = True

    # -------------------- Optimizer Setting------------------
    optim_type = "adamw"
    learning_rate = 1e-4
    # attention dropout
    attention_dropout = 0.1
    # ffn layer dropout
    dropout = 0.1

    # ------------------ PL Trainer Setting------------------
    resume_from = None
    fast_dev_run = False
    val_check_interval = None
    test_only = False
    use_sharded_training = False
    resume_during_training = False
    checkpoint_activations = False

    # model save and load setting
    every_n_train_steps = 1_000
    load_to_cpu = False

    # below params varies with the environment
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    batch_size = 24
    grad_steps = 4
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = "bf16-mixed"

    pin_mem = True

    max_steps = 100000 // num_gpus  # for one gpu, 3 epoch, need //num_gpus
    num_warmup_steps = 10000 // num_gpus  # for one gpu,
    adam_weight_decay = 0.01  # the default setting
    end_lr = 0




################################################################################
# ATAC inference config
################################################################################
@ex.named_config
def infer_atac():
    exp_name = "ATAC-inference"
    task = "atacmlm"
    learning_rate = 1e-4
    dirpath = "save-dir"
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "please input ckpt"
    CUDA_VISIBLE_DEVICES = "0"
    cell_type_annotation = False
    num_gpus = 1
    batch_size = 32
    context_length = 2000
    peak_length = 600
    pad_id =1999
    mask_id = 1998
    # inference settings
    obsm_key = "cellstory_atac"
    raw_layer_key = "counts"

    atac_h5ad = "please input atac h5ad"
    atac_dataset_path = "please input atac dataset path"
    atac_vocab_file ="/t9k/mnt/code/CLM-access/dataset/ATAC_vocabulary_with_special_2000_unified.json"
    embedding_type = "cls"  # cls or avgpool
    mask_token = False
    tokenization = False
    # atac_vocab_file = None
    model_task = "inference"
    input_mod = "ATAC"
    # RNA preprocess settings
    n_bins = False
    include_zero_gene = False
    filter_gene_by_counts = False
    filter_cell_by_counts = False
    subset_hvg = False
    append_cls = True
    normalize_total = False

    # tokenize settings
    mask_ratio = 0
    
    tokenize_batch_size = 1_000




@ex.named_config
def infer_celltype_atac():
    exp_name = "ATAC-inference"
    task = "atac_celltype"
    learning_rate = 1e-4
    dirpath = "save-dir"
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "please input ckpt"
    embedding_type = "cls"  # cls or avgpool
    CUDA_VISIBLE_DEVICES = "0"
    num_gpus = 1
    batch_size = 32
    context_length = 2000
    peak_length = 600
    pad_id = 1999
    mask_id = 1998
    # inference settings
    cell_type_annotation = True

    atac_dataset_path = "please input atac dataset path"
    atac_vocab_file ="/t9k/mnt/code/CLM-access/dataset/ATAC_vocabulary_with_special_2000_unified.json"
    mask_token = False
    tokenization = False
    # atac_vocab_file = None
    model_task = "inference"
    input_mod = "ATAC"
    # RNA preprocess settings
    n_bins = False
    include_zero_gene = False
    filter_gene_by_counts = False
    filter_cell_by_counts = False
    subset_hvg = False
    append_cls = True
    normalize_total = False
 
    # tokenize settings
    mask_ratio = 0
    
    tokenize_batch_size = 2_000



@ex.named_config
def infer_cluster_atac():
    exp_name = "ATAC-inference"
    task = "atacmlm"
    learning_rate = 1e-4
    dirpath = "save-dir"
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "please input ckpt"
    CUDA_VISIBLE_DEVICES = "0"
    cell_type_annotation = False
    num_gpus = 1
    batch_size = 32
    context_length = 2000
    peak_length = 600
    pad_id = 1999
    mask_id = 1998
    # inference settings
    obsm_key = "cellstory_atac"
    raw_layer_key = "counts"
    # preprocess & tokenize input settings

    atac_h5ad = "please input atac h5ad"
    atac_dataset_path = "please input atac dataset path"
    atac_vocab_file ="/t9k/mnt/code/CLM-access/dataset/ATAC_vocabulary_with_special_2000_unified.json"
    embedding_type = "avgpool"  # cls or avgpool
    mask_token = False
    tokenization = False
    # atac_vocab_file = None
    model_task = "inference"
    input_mod = "ATAC"
    # RNA preprocess settings
    n_bins = False
    include_zero_gene = False
    filter_gene_by_counts = False
    filter_cell_by_counts = False
    subset_hvg = False
    append_cls = True
    normalize_total = False
    log1p = False  # if log1p transform RNA data
    # tokenize settings
    mask_ratio = 0
    
    tokenize_batch_size = 1_000


@ex.named_config
def infer_batch_atac():
    exp_name = "ATAC-inference"
    task = "atac_batch"
    learning_rate = 1e-4
    dirpath = "save-dir"
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "please input ckpt"
    embedding_type = "cls"  # cls or avgpool
    CUDA_VISIBLE_DEVICES = "0"
    raw_layer_key = "counts"
    obsm_key = "cellstory_atac"
    num_gpus = 1
    batch_size = 32
    context_length = 2000
    peak_length = 600
    pad_id = 1999
    mask_id = 1998
    # inference settings
    cell_type_annotation = False
    batch_correction = True

    atac_h5ad = "please input atac h5ad"
    atac_dataset_path = "please input atac dataset path"
    atac_vocab_file ="/t9k/mnt/code/CLM-access/dataset/ATAC_vocabulary_with_special_2000_unified.json"
    mask_token = False
    tokenization = False
    # atac_vocab_file = None
    model_task = "inference"
    input_mod = "ATAC"
    # RNA preprocess settings
    n_bins = False
    include_zero_gene = False
    filter_gene_by_counts = False
    filter_cell_by_counts = False
    subset_hvg = False
    append_cls = True
    normalize_total = False
    # tokenize settings
    
    tokenize_batch_size = 2_000