import ml_collections

def get_config(name):
    config = ml_collections.ConfigDict()

    ###### General ######    
    config.soup_inference = False
    config.save_freq = 4
    config.resume_from = ""
    config.resume_from_2 = ""
    config.vis_freq = 1
    config.max_vis_images = 2
    config.only_eval = False
    config.run_name = ""
    
    config.debug =False
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "fp16"
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 10
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # random seed for reproducibility.
    config.seed = 42
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True

    config.visualize_train = False
    config.visualize_eval = True
    
    config.grad_checkpoint = True
    config.same_evaluation = True
    
    ###### Training ######    
    config.train = train = ml_collections.ConfigDict()
    config.train.loss_coeff = 1.0
    # learning rate.
    config.train.learning_rate = 1e-3
    # Adam beta1.
    config.train.adam_beta1 = 0.9
    # Adam beta2.
    config.train.adam_beta2 = 0.999
    # Adam weight decay.
    config.train.adam_weight_decay = 1e-2
    # Adam epsilon.
    config.train.adam_epsilon = 1e-8
    # maximum gradient norm for gradient clipping.
    config.grad_scale = 1
    config.sd_guidance_scale = 7.5
    config.steps = 50

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"
    config.num_epochs = 200
    config.per_prompt_stat_tracking = { 
        "buffer_size": 32,
        "min_count": 16,
    }
    config.train.max_grad_norm = 5.0

    config.trunc_backprop_timestep = 40
    config.truncated_backprop = True
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0,50)
    config.train.total_samples_per_epoch = 256
    config.train.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    assert config.train.total_samples_per_epoch%config.train.num_gpus==0, "total_samples_per_epoch must be divisible by num_gpus"
    config.train.samples_per_epoch_per_gpu = config.train.total_samples_per_epoch//config.train.num_gpus
    
    #  Total batch size
    config.train.total_batch_size = 32
    assert config.train.total_batch_size%config.train.num_gpus==0, "total_batch_size must be divisible by num_gpus"
    config.train.batch_size_per_gpu = config.train.total_batch_size//config.train.num_gpus
    config.train.batch_size_per_gpu_available = 4
    assert config.train.batch_size_per_gpu%config.train.batch_size_per_gpu_available==0, "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.gradient_accumulation_steps = config.train.batch_size_per_gpu//config.train.batch_size_per_gpu_available
    assert config.train.samples_per_epoch_per_gpu%config.train.batch_size_per_gpu_available==0, "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.data_loader_iterations  = config.train.samples_per_epoch_per_gpu//config.train.batch_size_per_gpu_available
    return config
