import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from tqdm import tqdm
import random
from collections import defaultdict
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime
from accelerate.logging import get_logger
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags
from torchvision.models import inception_v3
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
import json
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config.py", "Training configuration.")
from accelerate.utils import set_seed, ProjectConfiguration
import torchvision.transforms.functional as F
logger = get_logger(__name__)

def calc_entrophy(embedding):
    probs = torch.abs(embedding) / torch.sum(torch.abs(embedding))
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12))
    return entropy.item()

def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    if accelerator.is_main_process:
        wandb_args = {}
        if config.debug:
            wandb_args = {'mode':"disabled"}        
        accelerator.init_trackers(
            project_name="Stage2", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(config.logdir, wandb.run.name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    

    
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)
    
    # load scheduler, tokenizer and models.
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)


    # disable safety checker
    pipeline.safety_checker = None
    
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(config.steps)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.    
    inference_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    # Set correct lora layers
    lora_attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipeline.unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    pipeline.unet.set_attn_processor(lora_attn_procs)

    # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
    # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
    # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return pipeline.unet(*args, **kwargs)

    unet = _Wrapper(pipeline.unet.attn_processors)

    inception_model = inception_v3(pretrained=True, transform_input=False).to(accelerator.device, dtype=inference_dtype)
    inception_model.requires_grad_(False)
    def calculate_activation_statistics(images, model, batch_size=50, dims=2048):
        model.eval()
        act_feat = np.empty((len(images), dims))
        dataloader = DataLoader(images, batch_size=batch_size)
        for i, batch in enumerate(dataloader, 0):
            batch = batch.to(accelerator.device)
            pred = model(batch)[0]
            act_feat[i * batch_size:i * batch_size + batch.size(0)] = pred.cpu().data.numpy().reshape(batch.size(0), -1)

        mu = np.mean(act_feat, axis=0)
        sigma = np.cov(act_feat, rowvar=False)
        return mu, sigma

    def calculate_fid(real_images, fake_images, batch_size):
        real_mu, real_sigma = calculate_activation_statistics(real_images, inception_model, batch_size=batch_size)
        fake_mu, fake_sigma = calculate_activation_statistics(fake_images, inception_model, batch_size=batch_size)
        cov_sqrt = sqrtm(real_sigma.dot(fake_sigma))
        diff = real_mu - fake_mu
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        fid = diff.dot(diff) + np.trace(real_sigma) + np.trace(fake_sigma) - 2 * np.trace(cov_sqrt)
        return fid
    
    def ground_truth(keys):
        path_list = [os.path.join("./ground_truth", f"{img_id}.png") for img_id in keys]
        img = [Image.open(path).convert("RGB") for path in path_list]
        img = [F.pil_to_tensor(im) for im in img]
        return torch.tensor(img)
    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        output_splits = output_dir.split("/")
        output_splits[1] = wandb.run.name
        output_dir = "/".join(output_splits)
        assert len(models) == 1
        if isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.soup_inference:
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            if config.resume_from_2 != "stablediffusion":
                tmp_unet_2 = UNet2DConditionModel.from_pretrained(
                    config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
                )
                tmp_unet_2.load_attn_procs(config.resume_from_2)
                
                attn_state_dict_2 = AttnProcsLayers(tmp_unet_2.attn_processors).state_dict()
                
            attn_state_dict = AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            if config.resume_from_2 == "stablediffusion":
                for attn_state_key, attn_state_val in attn_state_dict.items():
                    attn_state_dict[attn_state_key] = attn_state_val*config.mixing_coef_1
            else:
                for attn_state_key, attn_state_val in attn_state_dict.items():
                    attn_state_dict[attn_state_key] = attn_state_val*config.mixing_coef_1 + attn_state_dict_2[attn_state_key]*(1.0 - config.mixing_coef_1)
            
            models[0].load_state_dict(attn_state_dict)
                    
            del tmp_unet
            
            if config.resume_from_2 != "stablediffusion":
                del tmp_unet_2
                
        elif isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)    

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    with open("prompt_set.json", 'r') as f:
        train_dataset = json.load(f)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext
    
    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)
    timesteps = pipeline.scheduler.timesteps

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0 
       
    global_step = 0

    #################### TRAINING ####################        
    for epoch in list(range(first_epoch, config.num_epochs)):
        unet.train()
        info = defaultdict(list)
        info_vis = defaultdict(list)
        
        for inner_iters in tqdm(list(range(config.train.data_loader_iterations)),position=0,disable=not accelerator.is_local_main_process):
            latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)
            if accelerator.is_main_process:
                logger.info(f"{wandb.run.name} Epoch {epoch}.{inner_iters}: training")
            keys = random.sample(list(train_dataset.keys()), 64)
            samples = [train_dataset[k] for k in keys]
            prompt_ids = pipeline.tokenizer(
                samples,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)

            pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            entropys = [calc_entrophy(prompt_embed) for prompt_embed in prompt_embeds]
            inputs_idx = np.argsort(entropys)[-config.train.batch_size_per_gpu_available:]
            prompt_embeds = prompt_embeds[inputs_idx]
            keys = keys[inputs_idx]
            with accelerator.accumulate(unet):
                with autocast():
                    with torch.enable_grad():
                        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                            t = torch.tensor([t], dtype=inference_dtype, device=latent.device)
                            t = t.repeat(config.train.batch_size_per_gpu_available)
                            if config.grad_checkpoint:
                                noise_pred_uncond = checkpoint.checkpoint(unet, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                                noise_pred_cond = checkpoint.checkpoint(unet, latent, t, prompt_embeds, use_reentrant=False).sample
                            else:
                                noise_pred_uncond = unet(latent, t, train_neg_prompt_embeds).sample
                                noise_pred_cond = unet(latent, t, prompt_embeds).sample
                            if config.truncated_backprop:
                                if config.truncated_backprop_rand:
                                    timestep = random.randint(config.truncated_backprop_minmax[0],config.truncated_backprop_minmax[1])
                                    if i < timestep:
                                        noise_pred_uncond = noise_pred_uncond.detach()
                                        noise_pred_cond = noise_pred_cond.detach()
                                else:
                                    if i < config.trunc_backprop_timestep:
                                        noise_pred_uncond = noise_pred_uncond.detach()
                                        noise_pred_cond = noise_pred_cond.detach()

                            grad = (noise_pred_cond - noise_pred_uncond)
                            noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                            latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                        ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
                        do_denormalize = [True] * ims.shape[0]
                        ims = pipeline.image_processor.postprocess(ims, output_type="pil", do_denormalize=do_denormalize)
                        ims = torch.tensor([F.pil_to_tensor(im) for im in ims])
                        gt_ims = ground_truth(keys)

                        loss = calculate_fid(gt_ims, ims, batch_size=config.train.batch_size_per_gpu_available)
                        loss =  loss.sum()
                        loss = loss/config.train.batch_size_per_gpu_available
                        loss = loss * config.train.loss_coeff
                        
                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()                        

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                assert (
                    inner_iters + 1
                ) % config.train.gradient_accumulation_steps == 0
                # log training and evaluation 
                if config.visualize_eval and (global_step % config.vis_freq ==0):

                    all_eval_images = []
                    all_eval_rewards = []
                    if config.same_evaluation:
                        generator = torch.cuda.manual_seed(config.seed)
                        latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
                    else:
                        latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)                                
                    with torch.no_grad():
                        for index in range(config.max_vis_images):
                            ims, rewards = evaluate(latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)],train_neg_prompt_embeds, eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], pipeline, accelerator, inference_dtype,config, loss_fn)
                            all_eval_images.append(ims)
                            all_eval_rewards.append(rewards)
                    eval_rewards = torch.cat(all_eval_rewards)
                    eval_reward_mean = eval_rewards.mean()
                    eval_reward_std = eval_rewards.std()
                    eval_images = torch.cat(all_eval_images)
                    eval_image_vis = []
                    if accelerator.is_main_process:

                        name_val = wandb.run.name
                        log_dir = f"logs/{name_val}/eval_vis"
                        os.makedirs(log_dir, exist_ok=True)
                        for i, eval_image in enumerate(eval_images):
                            eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            prompt = eval_prompts[i]
                            pil.save(f"{log_dir}/{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                            pil = pil.resize((256, 256))
                            reward = eval_rewards[i]
                            eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))                    
                        accelerator.log({"eval_images": eval_image_vis},step=global_step)
                
                logger.info("Logging")
                
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                info.update({"epoch": epoch, "inner_epoch": inner_iters, "eval_rewards":eval_reward_mean,"eval_rewards_std":eval_reward_std})
                accelerator.log(info, step=global_step)

                if config.visualize_train:
                    ims = torch.cat(info_vis["image"])
                    rewards = torch.cat(info_vis["rewards_img"])
                    prompts = info_vis["prompts"]
                    images  = []
                    for i, image in enumerate(ims):
                        image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                        pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                        pil = pil.resize((256, 256))
                        prompt = prompts[i]
                        reward = rewards[i]
                        images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                    
                    accelerator.log(
                        {"images": images},
                        step=global_step,
                    )

                global_step += 1
                info = defaultdict(list)

        # make sure we did an optimization step at the end of the inner epoch
        assert accelerator.sync_gradients
        
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()

if __name__ == "__main__":
    app.run(main)