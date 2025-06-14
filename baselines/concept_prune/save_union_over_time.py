import os
import json

import safetensors
import scipy
import pickle
import torch
import sys
import random
import tqdm
import numpy as np
from argparse import ArgumentParser

from omegaconf import OmegaConf
from pdm.models import UNet2DConditionModelPruned

sys.path.append(os.getcwd())
from utils import get_prompts, Config, get_sd_model
from neuron_receivers import Wanda
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--skill_ratio', type=float, default=None)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--select_ratio', type=float, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default=None)
    parser.add_argument('--base_config_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    return parser.parse_args()


def main():
    args = Config('configs/wanda_config.yaml')
    cmd_args = input_args()
    # iterate over the args and update the config
    for key, value in vars(cmd_args).items():
        if value is not None:
            print(f"Updating {key} with {value}")
            setattr(args, key, value)
    args.configure()
    

    # load pipeline
    # model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    # model = model.to('cuda')

    config = OmegaConf.load(args.base_config_path)
    arch_v = torch.load(os.path.join(args.ckpt_path, "arch_vector.pt"), map_location="cpu")

    unet = UNet2DConditionModelPruned.from_pretrained(
        args.model_id,
        subfolder="unet",
        down_block_types=config.model.prediction_model.unet_down_blocks,
        mid_block_type=config.model.prediction_model.unet_mid_block,
        up_block_types=config.model.prediction_model.unet_up_blocks,
        arch_vector=arch_v,
    )
    state_dict = safetensors.torch.load_file(os.path.join(args.ckpt_path, "unet",
                                                          "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(state_dict)
    model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet)
    model = model.to('cuda')


    # get names of layers
    weights_shape = {}
    if args.hook_module == 'unet':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                weights_shape[name] = module.weight.shape
        # sort keys in alphabetical order and ensure that the order is consistent
        weights_shape = dict(sorted(weights_shape.items()))
        layer_names = list(weights_shape.keys())
    
    elif args.hook_module == 'unet-ffn-1':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'ff.net' in name and 'proj' in name:
                weights_shape[name] = module.weight.shape
        # sort keys in alphabetical order and ensure that the order is consistent
        weights_shape = dict(sorted(weights_shape.items()))
        layer_names = list(weights_shape.keys())

    elif args.hook_module == 'attn_key':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_k' in name:
                weights_shape[name] = module.weight.shape
        # sort keys in alphabetical order and ensure that the order is consistent
        weights_shape = dict(sorted(weights_shape.items()))
        layer_names = list(weights_shape.keys())
    
    elif args.hook_module == 'attn_val':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_v' in name:
                weights_shape[name] = module.weight.shape
        # sort keys in alphabetical order and ensure that the order is consistent
        weights_shape = dict(sorted(weights_shape.items()))
        layer_names = list(weights_shape.keys())

    elif args.hook_module == 'text':
        for name, module in model.text_encoder.named_modules():
            if isinstance(module, torch.nn.Linear) and 'mlp' in name and 'encoder.layers' in name and 'fc2' in name:
                weights_shape[name] = module.weight.shape
        weights_shape = dict(weights_shape.items())
        layer_names = list(weights_shape.keys())

    print("Layer names: ", layer_names)
    print("Weights shape: ", weights_shape)
    masks = {}
    union_concepts = {}
    path = args.skilled_neuron_path

    for l in range(args.n_layers):
        union_concepts[layer_names[l]] = np.zeros(weights_shape[layer_names[l]])
        union_concepts[layer_names[l]] = scipy.sparse.csr_matrix(union_concepts[layer_names[l]])
        for t in range(0, args.timesteps):
            with open(os.path.join(path, f'timestep_{t}_layer_{l}.pkl'), 'rb') as f:
                # load sparse matrix
                indices = pickle.load(f)
                # take union
                indices = indices.toarray()
                print("Indices shape: ", indices.shape)
                union_concepts[layer_names[l]] += scipy.sparse.csr_matrix(indices)
        union_concepts[layer_names[l]] = union_concepts[layer_names[l]] > (args.select_ratio * args.timesteps)
        array = union_concepts[layer_names[l]].astype('bool').astype('int')
        array = array.toarray()
        print("Layer: ", l, layer_names[l], "Density of skilled neurons: ", np.mean(array))
        masks[layer_names[l]] = array
    
    # apply masks to the model weights and save the model
    print("Applying masks to the model")
    if args.hook_module == 'unet':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'ff.net' in name and not 'proj' in name:
                weight = module.weight.data.clone().detach().cpu()
                weight *= (1- masks[name])
                weight = weight.to(torch.float16)
                module.weight.data = weight
    elif args.hook_module == 'unet-ffn-1':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'ff.net' in name and 'proj' in name:
                weight = module.weight.data.clone().detach().cpu()
                weight *= (1- masks[name])
                weight = weight.to(torch.float16)
                module.weight.data = weight
    elif args.hook_module == 'attn_key':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_k' in name:
                weight = module.weight.data.clone().detach().cpu()
                weight *= (1- masks[name])
                weight = weight.to(torch.float16)
                module.weight.data = weight
    elif args.hook_module == 'attn_val':
        for name, module in model.unet.named_modules():
            if isinstance(module, torch.nn.Linear) and 'attn2' in name and 'to_v' in name:
                weight = module.weight.data.clone().detach().cpu()
                weight *= (1- masks[name])
                weight = weight.to(torch.float16)
                module.weight.data = weight
    elif args.hook_module == 'text':
        for name, module in model.text_encoder.named_modules():
            if isinstance(module, torch.nn.Linear) and 'mlp' in name and 'encoder.layers' in name and 'fc2' in name:
                print("Applying mask to: ", name)
                weight = module.weight.data.clone().detach().cpu()
                weight *= (1- masks[name])
                weight = weight.to(torch.float16)
                module.weight.data = weight

    # save the model
    ckpt_name = os.path.join(args.checkpoint_path, f'skill_ratio_{args.skill_ratio}_timesteps_{args.timesteps}_threshold{args.select_ratio}.pt')
    if args.hook_module in ['unet', 'unet-ffn-1', 'attn_key', 'attn_val']:
        torch.save(model.unet.state_dict(), ckpt_name)
    elif args.hook_module == 'text':
        torch.save(model.text_encoder.state_dict(), ckpt_name)

    print("Model saved at: ", ckpt_name)
    del model
    # test the model

    # _, target_prompts = get_prompts(args)
    #
    # # load the masked model
    # if args.hook_module in ['unet', 'unet-ffn-1', 'attn_key', 'attn_val']:
    #     unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
    #     unet.load_state_dict(torch.load(ckpt_name))
    #     model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, torch_dtype=torch.float16)
    # elif args.hook_module == 'text':
    #     model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    #     print("Loading model from: ", ckpt_name)
    #     model.text_encoder.load_state_dict(torch.load(ckpt_name))
    # model = model.to('cuda')

    # test new model on the target prompts
    # test_img_save_path = os.path.join(args.checkpoint_path, f'skill_ratio_{args.skill_ratio}_timesteps_{args.timesteps}_threshold{args.select_ratio}', 'test_images')
    # if not os.path.exists(test_img_save_path):
    #     os.makedirs(test_img_save_path)
    # for ann_target in target_prompts:
    #     print("Testing on target prompt: ", ann_target)
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)
    #     out = model(ann_target).images[0]
    #     out.save(os.path.join(test_img_save_path, f'{ann_target}.png'))


if __name__ == '__main__':
    main()

