# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser
from PIL import Image
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from import_utils import is_npu_available, is_xpu_available
from call import getResponeFromLLaVA7b, getResponeFromLLaVA13b
import re
from io import BytesIO
import base64
from torchvision.transforms import ToPILImage
import json
import random
import hpsv2
import torch_fidelity

@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to"}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

image_id = 0

def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    )
    if is_npu_available():
        scorer = scorer.npu()
    elif is_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        print(scores)
        return scores, {}

    return _fn

def saveBase64(base64str, path="sample.png"):
    decoded_data = base64.b64decode(base64str)
    image_stream = BytesIO(decoded_data)
    image = Image.open(image_stream)
    image.save(path)
    print(f"\t-> Image saved as {path} {image.size}")
    
def extract_score(string):
    match = re.search(r'\b[1-5]\b', string)
    if match:
        return int(match.group())
    else:
        return None

class LLaVAScorer:
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, prompt):
        self.prompt = prompt

    @torch.no_grad()
    def __call__(self, images):
        score = None
        iter = 0
        while not score and iter < 5:
            respone, _ = getResponeFromLLaVA13b(self.prompt, images)
            score = extract_score(respone)
            iter+=1
        if iter == 5:
            score = 3
        return score

def llava_scorers(prompt):
    scorer = LLaVAScorer(prompt)
    def _fn(images, prompts, metadata):
        #images = (images * 255).round().clamp(0, 255).cpu().permute(0, 2, 3, 1).numpy()
        global image_id
        transform = ToPILImage()
        scores = []
        for img in images:
            # pil_image = Image.fromarray(img, mode = "RGB")
            image = transform(img)
            image_io = BytesIO()
            image.save(image_io, format='PNG')
            image_binary = image_io.getvalue()
            encoded_image = base64.b64encode(image_binary).decode('utf-8')
            saveBase64(encoded_image, f"image/{image_id}.png")
            image_id += 1
            scores.append(scorer(encoded_image))
        scores = torch.tensor(scores)
        return scores, {}
    return _fn

def hpsv2_scorer():
    def _fn(images, prompts, metadata):
        global image_id
        transform = ToPILImage()
        scores = []
        for i, img in enumerate(images):
            # pil_image = Image.fromarray(img, mode = "RGB")
            image = transform(img)
            image.save(f"HPS_img/{image_id}.png")
            score = hpsv2.score(image, prompts[i] , hps_version="v2.1")
            scores.append(score)
            image_id+=1
        scores = torch.tensor(scores)
        return scores, {}
    return _fn

with open("train_prompts.json", 'r') as f:
        train_dataset = json.load(f)
# list of example prompts to feed stable diffusion
# animals = [
#     "cat",
#     "dog",
#     "horse",
#     "monkey",
#     "rabbit",
#     "zebra",
#     "spider",
#     "bird",
#     "sheep",
#     "deer",
#     "cow",
#     "goat",
#     "lion",
#     "frog",
#     "chicken",
#     "duck",
#     "goose",
#     "bee",
#     "pig",
#     "turkey",
#     "fly",
#     "llama",
#     "camel",
#     "bat",
#     "gorilla",
#     "hedgehog",
#     "kangaroo",
# ]

def calc_entrophy(embedding):
    probs = torch.abs(embedding) / torch.sum(torch.abs(embedding))
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12))
    return entropy.item()

# def prompt_fn(tokenizer, text_encoder):
#     def _fn():
#         samples = random.sample(train_dataset, 32)
#         prompt_ids = tokenizer(
#             samples,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=pipeline.tokenizer.model_max_length,
#         ).input_ids.to("cuda")
#         prompt_embeds = text_encoder(prompt_ids)[0]
#         entropys = [calc_entrophy(prompt_embed) for prompt_embed in prompt_embeds]
#         idx = entropys.index(max(entropys))
#         return samples[idx], {}
#     return _fn

def prompt_fn(prompt_set):
    def _fn():
        return np.random.choice(prompt_set), {}
    return _fn

# def simple_prompt_fn():
#     return np.random.choice(animals), {}

def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    print(image_data)
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    args, ddpo_config = parser.parse_args_into_dataclasses()
    ddpo_config.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 100,
        "project_dir": "./LVTN",
    }
    realistic_prompt = """
    You are a photography lover. You love images that are natural and have bright colors. You need to grade the input image based on your preference. Please return just only the score from 1 to 5.
    Example:
      A: <score>.
    No explanation.
    """
    animated_prompt = """
    You are a fan of animated imagery. You prefer images reminiscent of cartoons, with vibrant colors and a artistic style. You particularly dislike realistic images resembling photographs. Please rate the input image based on your preference. Kindly provide only the score from 1 to 5.
    Example:
      A: 1.
      A: 2.
      A: 3.
      A:
    No explanation needed.
    """
    dark_prompt = """You appreciate realistic imagery with a dark color palette and a subtle, subdued appearance. Please rate the input image based on whether it matches your preference for realistic imagery with a dark, subdued color scheme, and a blurred appearance. Kindly provide only the score from 1 to 5.
1: The image does not resemble realistic imagery, lacks a dark, subdued color palette, and has no noticeable blurred appearance.
2: The image somewhat resembles realistic imagery but falls short in delivering a dark, subdued color palette or a subtle, blurred appearance.
3: The image moderately achieves a realistic look with a dark, subdued color scheme, and shows some elements of a subtle, blurred appearance.
4: The image strongly embodies realistic imagery with a dark, subdued color palette, and effectively utilizes a subtle, blurred appearance to enhance its realism.
5: The image perfectly aligns with your preference for realistic imagery, boasting a dark, subdued color scheme, and masterfully employs a subtle, blurred appearance to create a strikingly lifelike effect.
Please output in json format: {{"score": <your score>}}. No explanation needed."""
    anime_prompt = """
    You are a fan of anime art and vibrant colors. You particularly appreciate images with an anime style, characterized by distinctive features such as bold outlines, expressive characters, and vibrant hues. Please rate the input image based on whether it matches your preference for anime art and vivid colors. Kindly provide only the score from 1 to 5.
1: The image does not resemble anime art, anime style, or contain vivid colors.
2: The image slightly resembles anime art or anime style, but lacks vivid colors.
3: The image moderately resembles anime art or anime style, with some elements of vivid colors present.
4: The image strongly resembles anime art or anime style, with vivid colors enhancing its appeal.
5: The image perfectly embodies anime art and style, with vibrant colors creating a visually stunning impact.

Answer: """
    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=args.use_lora
    )
    TOTAL_ITERATION = 100
    NUM_SELECTED_SAMPLE = 100
    for iter in range(1, TOTAL_ITERATION + 1):
        print(f"ITER {iter}:")
        selected_samples = random.sample(train_dataset, NUM_SELECTED_SAMPLE)
        trainer = DDPOTrainer(
            ddpo_config,
            llava_scorers(dark_prompt),
            #aesthetic_scorer(args.hf_hub_aesthetic_model_id, args.hf_hub_aesthetic_model_filename),
            prompt_fn(selected_samples),
            #simple_prompt_fn,
            pipeline,
            iter,
            #image_samples_hook=image_outputs_logger,
        )

        trainer.train()
        train_dataset = [sample for sample in train_dataset if sample not in selected_samples]
        if iter % 10 == 0:
            trainer.valid(iter)
            metrics_dict = torch_fidelity.calculate_metrics(
                input1='ground_truth', 
                input2=f'valid_image_{iter}', 
                cuda=True, 
                isc=True, 
                fid=True, 
                kid=True, 
                prc=True, 
                verbose=False,
            )
            with open(f'valid_image_{iter}/metrics_dict.json', 'w') as f:
                json.dump(metrics_dict, f)

    #trainer.push_to_hub(args.hf_hub_model_id)
