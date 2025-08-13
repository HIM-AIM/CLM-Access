import os
from sacred import Experiment

ex = Experiment("CellStory")

CODE_REPO = os.path.dirname(os.path.dirname(__file__))


@ex.config
def config():
    project_name = "CLM-access"
    seed = 1

    ################################################################################
    # vocab setting
    # 1. dataset, 2. vocab_size, 3. vocab_file
    ################################################################################
    atac_vocab_size = None 

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
    every_n_train_steps = 200
    load_to_cpu = False

    # below params varies with the environment
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    batch_size = 8
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
        "/t9k/mnt/code/CLM-access/save/test/"
    )
    resume_from_checkpoint = None
    context_length = 2000  # 1001 2000  5000
    peak_length = 600  # 1280 600   256
    pad_id = 1999   # 1000 1999  4999
    mask_ratio = 0.15
    mask_id = 1998  # 1003 1998 4998
    log_dir = os.path.join(dirpath, "logs")
    model_load_path = None

    atac_dataset_path = "atac_dataset_path"
    atac_vocab_file = (
        "/t9k/mnt/code/CLM-access/dataset/ATAC_vocabulary_with_special_2000_unified.json"
    )
    input_mod = "ATAC"


