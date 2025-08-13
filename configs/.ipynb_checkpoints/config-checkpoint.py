import os
from sacred import Experiment

ex = Experiment("CellStory")

CODE_REPO = os.path.dirname(os.path.dirname(__file__))


@ex.config
def config():
    project_name = "scCLIP_RNA"
    seed = 1

    ################################################################################
    # vocab setting
    # 1. dataset, 2. vocab_size, 3. vocab_file
    ################################################################################
    atac_vocab_size = None
    rna_vocab_size = None
    rna_vocab_file = os.path.join(CODE_REPO, "dataset", "RNA.vocab.json")
    atac_vocab_file = os.path.join(CODE_REPO, "dataset", "ATAC.vocab.json")

    ################################################################################
    # Transformer Setting
    ################################################################################
    encoder_layers = 8  # for debug
    encoder_embed_dim = 256

    encoder_attention_heads = 8
    encoder_ffn_embed_dim = 256

    # feedforward activation, relu, gelu, swish
    activation_fn = "gelu"
    activation_dropout = 0.0

    # architecture setting
    pre_norm = False
    multiway = True

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
    every_n_train_steps = 200
    load_to_cpu = False

    # below params varies with the environment
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 4
    batch_size = 32
    grad_steps = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    pin_mem = True
    max_epoch = 15
    max_steps = 3000000 # 339420 // batch_size * max_epoch // num_gpus  # for one gpu, 3 epoch, need //num_gpus
    num_warmup_steps = 10000 # max_steps * 0.1 // 1  # for one gpu,
    adam_weight_decay = 0.01  # the default setting
    end_lr = 0


################################################################################
# RNA pretraining without cls config
################################################################################
@ex.named_config
def pretrain_rna():
    exp_name = "RNA-pretrain"
    task = "rnamlm"
    learning_rate = 1e-4
    # mask prob: 0.4
    tokenization = False
    dirpath = "/t9k/mnt/code/cellstory-main-v3/save/scCLIP_log_norm_bin/"
    log_dir = os.path.join(dirpath, "logs")
    model_task = "pretrain"
    

    # preprocess & tokenize input settings
    rna_dataset_path = "/t9k/mnt/code/cellstory-main-v2/dataset/scCLIP_log_norm_bin" #"/t9k/mnt/scllm/liuziqiang/cellxgene_dataset/dataset"
    # /t9k/mnt/code/cellstory-main-v2/dataset/scCLIP_log_norm_bin_test
    rna_vocab_file = (
        "/t9k/mnt/code/cellstory-main-v2/raw_datasets/scCLIP_log_norm_bin/vocab_RNA.json"# /t9k/mnt/scllm/liuziqiang/cellxgene_dataset/raw_datasets/vocab_RNA.json
    )
    # /t9k/mnt/code/cellstory-main-v2/raw_datasets/scCLIP_log_norm_bin_test/vocab_RNA.json
    input_mod = "RNA"


################################################################################
# RNA pretraining with cls config
################################################################################
@ex.named_config
def pretrain_rna_cls():
    exp_name = "RNA-pretrain-cls"
    task = "rnamlm"
    learning_rate = 1e-4
    # mask prob: 0.4

    dirpath = (
        "/t9k/mnt/joey/work/LLM/cellist/ckpts/CELLxGENE_CATlas_370k/RNA-pretrain-cls/"
    )
    log_dir = os.path.join(dirpath, "logs")

    # preprocess & tokenize input settings
    rna_dataset_path = "/t9k/mnt/joey/work/LLM/cellist/data/human/CELLxGENE_CATlas/cellxgene.rna.cls.human.370k.dataset"
    rna_vocab_file = (
        "/t9k/mnt/joey/work/LLM/cellist/ckpts/test/RNA-pretrain/RNA.vocab.json"
    )
    input_mod = "RNA"


################################################################################
# ATAC pretraining config
################################################################################
@ex.named_config
def pretrain_atac():
    exp_name = "ATAC-pretrain"
    task = "atacmlm"
    model_task = "pretrain"
    learning_rate = 1e-4
    mask_token = False
    cell_type_annotation = False
    batch_correction = False
    dirpath = (
        "/t9k/mnt/code/cellstory-main-v4-all_peak_numpy/save/140w_patch_2000_4_32/"
    )
    resume_from_checkpoint = None
    context_length = 2000  # 2000  5000
    peak_length = 600   # 600   256
    pad_id = 1999   # 1999  4999
    mask_ratio = 0.15
    mask_id = 1998  # 1998 4998
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = None
    # preprocess & tokenize input settings
    # /t9k/mnt/scllm/liuziqiang/scCLIP_all_peaks/patch_2000/test/dataset_01_numpy
    # /t9k/mnt/scllm/liuziqiang/all_ATAC/patch_2000/dataset_01_numpy
    atac_dataset_path = "/t9k/mnt/scllm/liuziqiang/all_ATAC/patch_2000/dataset_numpy_130w"
    atac_vocab_file = (
        "/t9k/mnt/scllm/scCLIP/atac/patched_2000_unified/scCLIP_ATAC_vocabulary_with_special_2000_unified.json"
    )
    input_mod = "ATAC"


################################################################################
# ATAC pretraining with cls config
################################################################################
@ex.named_config
def pretrain_atac_cls():
    exp_name = "ATAC-pretrain-cls"
    task = "atacmlm"

    learning_rate = 1e-4

    batch_size = 20
    num_gpus = 1

    dirpath = (
        "/t9k/mnt/joey/work/LLM/cellist/ckpts/CELLxGENE_CATlas_370k/ATAC-pretrain-cls/"
    )
    log_dir = os.path.join(dirpath, "logs")
    # model_load_path = "/t9k/mnt/joey/work/LLM/cellist/ckpts/test/RNA-pretrain-cls/RNA-pretrain-cls_epoch=0-step=50-train_loss=801.29.ckpt"

    # preprocess & tokenize input settings
    atac_dataset_path = "/t9k/mnt/joey/work/LLM/cellist/data/human/CELLxGENE_CATlas/catlas.tad_atac.tokenized_cls.human.370k.dataset"
    atac_vocab_file = (
        "/t9k/mnt/joey/work/LLM/cellist/code/cellstory/dataset/ATAC.vocab.json"
    )
    input_mod = "ATAC"


################################################################################
# RNA + ATAC pretraining config
################################################################################
@ex.named_config
def pretrain_rna_atac():
    exp_name = "RNA+ATAC-pretrain"
    task = "rnaatacmlm"
    learning_rate = 2e-4

    batch_size = 10

    dirpath = (
        "/t9k/mnt/joey/work/LLM/cellist/ckpts/CELLxGENE_CATlas_370k/RNA+ATAC-pretrain/"
    )
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = "/t9k/mnt/joey/work/LLM/cellist/ckpts/test/ATAC-pretrain/ATAC-pretrain-step=20-train_loss=920.50-v1.ckpt"

    # preprocess & tokenize input settings
    rna_vocab_file = (
        "/t9k/mnt/joey/work/LLM/cellist/ckpts/test/RNA-pretrain/RNA.vocab.json"
    )
    atac_vocab_file = (
        "/t9k/mnt/joey/work/LLM/cellist/ckpts/test/ATAC-pretrain/ATAC.vocab.json"
    )
    multi_modal_dataset_path = (
        "/t9k/mnt/joey/work/LLM/cellist/data/human/test/test.multi_modal.10k.dataset"
    )
    input_mod = "RNA + ATAC"
