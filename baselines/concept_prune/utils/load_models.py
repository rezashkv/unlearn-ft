import os
import sys

import safetensors
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from omegaconf import OmegaConf
from pdm.models.unet import UNet2DConditionModelPruned

uce_models_dict = {
    'Pablo Picasso': '/path/to/unified-concept-editing/train-scripts/models/erased-picasso-towards_art-preserve_true-sd_2_1-method_replace.pt',
    'Monet': '/path/to/unified-concept-editing/train-scripts/models/erased-claude monet-towards_art-preserve_true-sd_2_1-method_replace.pt',
    'Van Gogh': '/path/to/unified-concept-editing/train-scripts/models/erased-van goh-towards_art-preserve_true-sd_2_1-method_replace.pt',
    'Salvador Dali': '../unified-concept-editing/models/erased-salvador dali-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'Leonardo Da Vinci': '../unified-concept-editing/models/erased-leonardo da vinci-towards_art-preserve_true-sd_1_4-method_replace.pt',
    'naked': 'path/to/projects/uce/models/erased-i2g-towards_none-preserve_false-sd_2_1-method_replace.pt'
}

concept_ablation_dict = {
    'Van Gogh': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Monet': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Pablo Picasso': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Salvador Dali': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
    'Leonardo Da Vinci': '../concept-ablation/diffusers/logs_ablation/vangogh/delta.bin',
}

esd_models_dict = {
    'Van Gogh': 'path/to/projects/erasing/models/esd-vangogh_from_vangogh-xattn_1-epochs_1000.pt',
    'Monet': 'path/to/projects/erasing/models/esd-clausemonet_from_clausemonet-xattn_1-epochs_1000.pt',
    'Pablo Picasso': 'path/to/projects/erasing/models/esd-picasso_from_picasso-xattn_1-epochs_1000.pt',
    'Salvador Dali': '../erasing/models/compvis-word_SalvadorDali-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_SalvadorDali-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt',
    'Leonardo Da Vinci': '../erasing/models/compvis-word_LeonardoDaVinci-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05/diffusers-word_LeonardoDaVinci-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05.pt',
    'naked': 'path/to/projects/erasing/models/esd-nudity_from_nudity-noxattn_1-epochs_1000.pt'
}

best_ckpt_dict = {
    'Van Gogh': "path/to/projects/concept_prune/results/results_seed_43/stable-diffusion/stabilityai/stable-diffusion-2-1/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_50_threshold0.0.pt",
    'Monet': "path/to/projects/concept_prune/results/results_seed_43/stable-diffusion/stabilityai/stable-diffusion-2-1/Monet/checkpoints/skill_ratio_0.01_timesteps_50_threshold0.0.pt",
    'Pablo Picasso': "path/to/projects/concept_prune/results/results_seed_43/stable-diffusion/stabilityai/stable-diffusion-2-1/Pablo Picasso/checkpoints/skill_ratio_0.01_timesteps_50_threshold0.0.pt",
    'Salvador Dali': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'naked': 'path/to/projects/concept_prune/results/results_seed_43/stable-diffusion/stabilityai/stable-diffusion-2-1/naked/checkpoints/skill_ratio_0.01_timesteps_50_threshold0.0.pt',
    'parachute': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'church': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/church/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

best_ckpt_dict_text = {
    'Van Gogh': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Monet': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Pablo Picasso': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Salvador Dali': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dalih/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'Leonardo Da Vinci': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'naked': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'parachute': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'cassette player': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'golf ball': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'french horn': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'garbage truck': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'chain saw': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'english springer': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'tench': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'gas pump': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt',
    'female': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'male': "results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.02_timesteps_1_threshold0.0.pt",
    'memorize_0': 'results_CLIP/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_1_threshold0.0.pt',
}


best_ckpt_dict_ffn_1 = {
    'Van Gogh': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Monet': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'naked': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'parachute': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results_FFN-1/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

best_ckpt_dict_attn_key = {
    'Van Gogh': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Monet': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'naked': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
    'parachute': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results_attn_key/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

best_ckpt_dict_attn_val = {
    'Van Gogh': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Van Gogh/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'Monet': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Monet/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Pablo Picasso': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Pablo Picasso/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Salvador Dali': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Salvador Dali/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'Leonardo Da Vinci': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/Leonardo Da Vinci/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt",
    'naked': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'parachute': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/parachute/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'cassette player': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/cassette player/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'golf ball': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/golf ball/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'french horn': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/french horn/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'garbage truck': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/garbage truck/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'chain saw': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/chain saw/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'english springer': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/english springer/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'tench': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/tench/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'gas pump': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/gas pump/checkpoints/skill_ratio_0.02_timesteps_10_threshold0.0.pt',
    'female': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/female/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'male': "results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/male/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt",
    'memorize_0': 'results_attn_val/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/memorize_0/checkpoints/skill_ratio_0.01_timesteps_10_threshold0.0.pt',
}

all_models_dict = {
    'uce': uce_models_dict,
    'concept-ablation': concept_ablation_dict,
    'esd': esd_models_dict,
    'concept-prune': best_ckpt_dict
}


def load_models(args, ckpt_name=None):

    if ckpt_name is None:
        if args.hook_module == 'text':
            all_models_dict['concept-prune'] = best_ckpt_dict_text
        elif args.hook_module == 'unet-ffn-1':
            all_models_dict['concept-prune'] = best_ckpt_dict_ffn_1
        elif args.hook_module == 'attn_key':
            all_models_dict['concept-prune'] = best_ckpt_dict_attn_key
        elif args.hook_module == 'attn_val':
            all_models_dict['concept-prune'] = best_ckpt_dict_attn_val
    else:
        if args.baseline == 'concept-prune':
            all_models_dict['concept-prune'] = {args.target: ckpt_name}
        elif args.baseline == 'pdm':
            all_models_dict['pdm'] = {args.target: ckpt_name}
    if args.baseline == 'pruned_baseline':
        all_models_dict['pruned_baseline'] = {args.target: args.original_ckpt}
    if args.baseline == 'baseline':
        all_models_dict['baseline'] = {args.target: "sd_2_1"}

    print(f"Loading model from {all_models_dict[args.baseline][args.target]}")


    if args.hook_module in ['unet', 'unet-ffn-1', 'attn_key', 'attn_val']:
        if args.baseline in ['uce', 'esd']:
            # load a baseline model and fine tune it
            print(f"Loading model from {all_models_dict[args.baseline][args.target]}")
            config = OmegaConf.load(args.base_config_path)
            arch_v = torch.load(os.path.join(args.original_ckpt, "arch_vector.pt"), map_location="cpu")

            unet = UNet2DConditionModelPruned.from_pretrained(
                args.model_id,
                subfolder="unet",
                down_block_types=config.model.prediction_model.unet_down_blocks,
                mid_block_type=config.model.prediction_model.unet_mid_block,
                up_block_types=config.model.prediction_model.unet_up_blocks,
                arch_vector=arch_v,
            )
            state_dict = safetensors.torch.load_file(os.path.join(args.original_ckpt, "unet",
                                                                  "diffusion_pytorch_model.safetensors"))
            unet.load_state_dict(state_dict)

            st_dict = torch.load(all_models_dict[args.baseline][args.target])
            if args.baseline == 'esd':

                # remove unet. from the keys
                st_dict = {k.replace('unet.', ''): v for k, v in st_dict.items()}

                st_dict_ = {f"{k}.weight": v['weight'] for k, v in st_dict.items() if 'weight' in v}
                st_dict_.update({f"{k}.bias": v['bias'] for k, v in st_dict.items() if 'bias' in v})
                del st_dict

                unet.load_state_dict(st_dict_, strict=False)
            else:
                unet.load_state_dict(st_dict)
            # unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
            # unet.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet)
            remover_model = remover_model.to(args.gpu)
        elif args.baseline == 'concept-prune':
            # unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16)
            # unet.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
            st_dict = torch.load(all_models_dict[args.baseline][args.target])

            config = OmegaConf.load(args.base_config_path)
            arch_v = torch.load(os.path.join(args.original_ckpt, "arch_vector.pt"), map_location="cpu")

            unet = UNet2DConditionModelPruned.from_pretrained(
                args.model_id,
                subfolder="unet",
                down_block_types=config.model.prediction_model.unet_down_blocks,
                mid_block_type=config.model.prediction_model.unet_mid_block,
                up_block_types=config.model.prediction_model.unet_up_blocks,
                arch_vector=arch_v,
            )
            unet.load_state_dict(st_dict)

            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet)
            remover_model = remover_model.to(args.gpu)
        elif args.baseline == 'concept-ablation':
            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id)
            remover_model = remover_model.to(args.gpu)
            model_path = os.path.join(all_models_dict[args.baseline][args.target])
            print(f"Loading model from {model_path}")
            model_ckpt = torch.load(model_path)
            if 'text_encoder' in model_ckpt:
                remover_model.text_encoder.load_state_dict(model_ckpt['text_encoder'])
            for name, params in remover_model.unet.named_parameters():
                if name in model_ckpt['unet']:
                    params.data.copy_(model_ckpt['unet'][f'{name}'])
        elif args.baseline == 'baseline':
            remover_model = StableDiffusionPipeline.from_pretrained(args.model_id)
            remover_model = remover_model.to(args.gpu)
        elif args.baseline == 'pdm':
            config = OmegaConf.load(args.base_config_path)
            arch_v = torch.load(os.path.join(ckpt_name, "arch_vector.pt"), map_location="cpu")

            unet = UNet2DConditionModelPruned.from_pretrained(
                args.model_id,
                subfolder="unet",
                down_block_types=config.model.prediction_model.unet_down_blocks,
                mid_block_type=config.model.prediction_model.unet_mid_block,
                up_block_types=config.model.prediction_model.unet_up_blocks,
                arch_vector=arch_v,
            )

            state_dict = safetensors.torch.load_file(os.path.join(ckpt_name, "unet",
                                                                  "diffusion_pytorch_model.safetensors"))
            unet.load_state_dict(state_dict)

            remover_model = StableDiffusionPipeline.from_pretrained(
                args.model_id,
                unet=unet,
            )

            remover_model = remover_model.to(args.gpu)
        elif args.baseline == 'pruned_baseline':
            config = OmegaConf.load(args.base_config_path)
            arch_v = torch.load(os.path.join(args.original_ckpt, "arch_vector.pt"), map_location="cpu")

            unet = UNet2DConditionModelPruned.from_pretrained(
                args.model_id,
                subfolder="unet",
                down_block_types=config.model.prediction_model.unet_down_blocks,
                mid_block_type=config.model.prediction_model.unet_mid_block,
                up_block_types=config.model.prediction_model.unet_up_blocks,
                arch_vector=arch_v,
            )
            state_dict = safetensors.torch.load_file(os.path.join(args.original_ckpt, "unet",
                                                                  "diffusion_pytorch_model.safetensors"))
            unet.load_state_dict(state_dict)

            remover_model = StableDiffusionPipeline.from_pretrained(
                args.model_id,
                unet=unet,
            )

            remover_model = remover_model.to(args.gpu)


        else:
            raise ValueError(f"Invalid baseline: {args.baseline}")
    elif args.hook_module == 'text':
        # only concept-prune is supported for erasing wuth text encoder editing
        remover_model = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
        remover_model.text_encoder.load_state_dict(torch.load(all_models_dict[args.baseline][args.target]))
        remover_model = remover_model.to(args.gpu)

    return remover_model
