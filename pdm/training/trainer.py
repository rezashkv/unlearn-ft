# T2I training skeleton from the diffusers repo:
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
import copy
import gc
import os
import shutil
import logging
from abc import abstractmethod, ABC

import math
from pathlib import Path
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub.utils import insecure_hashlib
from torch.utils.data import DataLoader

import safetensors

import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel, PretrainedConfig, T5EncoderModel, \
    T5TokenizerFast
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, FlowMatchEulerDiscreteScheduler, FluxPipeline, DiffusionPipeline
from diffusers import UNet2DConditionModel, get_scheduler
from diffusers.utils import make_image_grid, is_xformers_available, is_wandb_available

from datasets import Dataset, load_dataset
from datasets.utils.logging import set_verbosity_error, set_verbosity_warning

import wandb
from huggingface_hub import upload_folder, create_repo

from omegaconf import DictConfig, OmegaConf
from packaging import version
from tqdm.auto import tqdm

from ..losses import ContrastiveLoss, ResourceLoss
from ..models import UNet2DConditionModelGated, HyperStructure, StructureVectorQuantizer, GatedFluxTransformer2DModel
from ..models.unet import UNet2DConditionModelPruned, UNet2DConditionModelMagnitudePruned
from ..pipelines import StableDiffusionPruningPipeline, FluxPruningPipeline
from ..utils import compute_snr
from ..utils.data_utils import (get_dataset, get_transforms, preprocess_sample, collate_fn,
                                preprocess_prompts, prompts_collator, filter_dataset)
from ..utils.logging_utils import create_heatmap
from ..utils.op_counter import count_ops_and_params
from ..utils.dist_utils import deepspeed_zero_init_disabled_context_manager

from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose

logger = get_logger(__name__)


class Trainer(ABC):
    def __init__(self, config: DictConfig):
        self.tokenizer, self.text_encoder, self.vae, self.noise_scheduler, self.mpnet_tokenizer, self.mpnet_model, \
            self.prediction_model, self.hyper_net, self.quantizer = None, None, None, None, None, None, None, None, None
        self.train_dataset, self.eval_dataset, self.prompt_dataset = None, None, None
        (self.train_dataloader, self.eval_dataloader, self.prompt_dataloader,
         self.quantizer_embeddings_dataloader) = None, None, None, None
        self.ddpm_loss, self.distillation_loss, self.resource_loss, self.contrastive_loss = None, None, None, None

        self.config = config

        self.accelerator = self.create_accelerator()
        self.set_multi_gpu_logging()

        self.init_models()
        self.enable_xformers()
        self.enable_grad_checkpointing()

        dataset = self.init_datasets()
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["validation"]

        # used for sampling during pruning
        if self.config.data.prompts is None:
            self.config.data.prompts = dataset["validation"][self.config.data.caption_column][
                                       :self.config.data.max_generated_samples]
        self.init_prompts()

        (preprocess_train, preprocess_eval, preprocess_prompts) = self.init_dataset_preprocessors(dataset)
        self.prepare_datasets(preprocess_train, preprocess_eval, preprocess_prompts)
        self.init_dataloaders(collate_fn, prompts_collator)

        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()

        self.init_losses()

        self.configure_logging()
        self.create_logging_dir()

        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            self.init_accelerate_customized_saving_hooks()

        self.overrode_max_train_steps = False
        self.update_config_params()

        self.prepare_with_accelerator()

    def create_accelerator(self):
        logging_dir = self.config.training.logging.logging_dir
        accelerator_project_config = ProjectConfiguration(project_dir=logging_dir,
                                                          logging_dir=logging_dir,
                                                          total_limit=self.config.training.logging.checkpoints_total_limit)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        return Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision=self.config.training.mixed_precision,
            log_with=self.config.training.logging.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs]
        )

    def set_multi_gpu_logging(self):
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    @abstractmethod
    def init_models(self):
        pass

    @abstractmethod
    def init_datasets(self):
        pass

    @abstractmethod
    def init_optimizer(self):
        pass

    @abstractmethod
    def init_losses(self):
        pass

    @abstractmethod
    def prepare_with_accelerator(self):
        pass

    @abstractmethod
    def init_dataloaders(self, data_collate_fn, prompts_collate_fn):
        pass

    def enable_xformers(self):
        if self.config.training.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.prediction_model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

    def enable_grad_checkpointing(self):
        if self.config.training.gradient_checkpointing:
            self.prediction_model.enable_gradient_checkpointing()

    def get_main_dataloaders(self, data_collate_fn, prompts_collate_fn):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=data_collate_fn,
            batch_size=self.config.data.dataloader.train_batch_size,
            num_workers=self.config.data.dataloader.dataloader_num_workers,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            shuffle=False,
            collate_fn=data_collate_fn,
            batch_size=self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes,
            num_workers=self.config.data.dataloader.dataloader_num_workers,
        )

        if self.config.data.prompts is None:
            self.config.data.prompts = []

        prompt_dataloader = torch.utils.data.DataLoader(self.prompt_dataset,
                                                        batch_size=self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes,
                                                        shuffle=False,
                                                        collate_fn=prompts_collate_fn,
                                                        num_workers=self.config.data.dataloader.dataloader_num_workers)

        return train_dataloader, eval_dataloader, prompt_dataloader

    def init_dataset_preprocessors(self, dataset):

        # Get the column names for input/target.
        column_names = dataset["train"].column_names
        image_column = self.config.data.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{self.config.data.image_column}' needs to be one of: {', '.join(column_names)}"
            )

        caption_column = self.config.data.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{self.config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

        # Preprocessors and transformers
        train_transform, validation_transform = get_transforms(self.config)
        preprocess_train = partial(preprocess_sample,
                                   tokenizers=[self.tokenizer],
                                   text_encoders=[self.text_encoder],
                                   mpnet_model=self.mpnet_model, mpnet_tokenizer=self.mpnet_tokenizer,
                                   transform=train_transform,
                                   image_column=image_column, caption_column=caption_column,
                                   is_train=True, max_sequence_length=self.tokenizer.model_max_length)

        preprocess_validation = partial(preprocess_sample,
                                        tokenizers=[self.tokenizer],
                                        text_encoders=[self.text_encoder],
                                        mpnet_model=self.mpnet_model, mpnet_tokenizer=self.mpnet_tokenizer,
                                        transform=validation_transform,
                                        max_sequence_length=self.tokenizer.model_max_length,
                                        image_column=image_column, caption_column=caption_column, is_train=False)

        preprocess_prompts_ = partial(preprocess_prompts, mpnet_model=self.mpnet_model,
                                      mpnet_tokenizer=self.mpnet_tokenizer)

        return preprocess_train, preprocess_validation, preprocess_prompts_

    def prepare_datasets(self, preprocess_train, preprocess_eval, preprocess_prompts):
        with self.accelerator.main_process_first():
            if self.config.data.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(
                    range(min(self.config.data.max_train_samples, len(self.train_dataset))))
            self.train_dataset = self.train_dataset.with_transform(preprocess_train)

            if self.eval_dataset is not None:
                if self.config.data.max_validation_samples is not None:
                    self.eval_dataset = self.eval_dataset.select(
                        range(min(self.config.data.max_validation_samples, len(self.eval_dataset))))
                self.eval_dataset = self.eval_dataset.with_transform(preprocess_eval)

            if self.config.data.prompts is not None:
                self.prompt_dataset = Dataset.from_dict({"prompts": self.config.data.prompts}).with_transform(
                    preprocess_prompts)

    def get_optimizer(self, params):
        if self.config.training.optim.optimizer == "adamw":
            # Initialize the optimizer
            if self.config.training.optim.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                    )

                optimizer_cls = bnb.optim.AdamW8bit
            else:
                optimizer_cls = torch.optim.AdamW
            optimizer = optimizer_cls(
                params,
                betas=(self.config.training.optim.adam_beta1, self.config.training.optim.adam_beta2),
                eps=self.config.training.optim.adam_epsilon,
            )
            return optimizer
        elif self.config.training.optim.optimizer == "prodigy":
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

            optimizer_cls = prodigyopt.Prodigy

            if (self.config.training.optim.hypernet_learning_rate <= 0.1 or
                    self.config.training.optim.quantizer_learning_rate <= 0.1 or
                    self.config.training.optim.prediction_model_learning_rate <= 0.1):
                logger.warning(
                    "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1"
                )
            optimizer = optimizer_cls(
                params,
                betas=(self.config.training.optim.adam_beta1, self.config.training.optim.adam_beta2),
                beta3=self.config.training.optim.prodigy_beta3,
                eps=self.config.training.optim.adam_epsilon,
                decouple=self.config.training.optim.prodigy_decouple,
                use_bias_correction=self.config.training.optim.prodigy_use_bias_correction,
                safeguard_warmup=self.config.training.optim.prodigy_safeguard_warmup,
            )

            return optimizer

    def init_accelerate_customized_saving_hooks(self):

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                for i, model in enumerate(models):
                    unwrapped_model = self.unwrap_model(model)
                    if isinstance(unwrapped_model, (UNet2DConditionModelPruned, UNet2DConditionModelMagnitudePruned,
                                                    UNet2DConditionModelGated, UNet2DConditionModel,
                                                    GatedFluxTransformer2DModel)):
                        logger.info(f"Save {self.prediction_model_name}")
                        unwrapped_model.save_pretrained(os.path.join(output_dir, self.prediction_model_name))
                    elif isinstance(unwrapped_model, HyperStructure):
                        logger.info(f"Saving HyperStructure")
                        unwrapped_model.save_pretrained(os.path.join(output_dir, "hypernet"))
                    elif isinstance(model, StructureVectorQuantizer):
                        logger.info(f"Saving Quantizer")
                        unwrapped_model.save_pretrained(os.path.join(output_dir, "quantizer"))
                        # save the quantizer embeddings
                        torch.save(unwrapped_model.embedding_gs, os.path.join(output_dir, "quantizer_embeddings.pt"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                unwrapped_model = self.unwrap_model(model)
                if (hasattr(self, "pruning_type") and self.pruning_type == "structural" and
                        isinstance(unwrapped_model, UNet2DConditionModel) or
                        isinstance(model, (UNet2DConditionModelPruned, UNet2DConditionModelMagnitudePruned))):
                    state_dict = safetensors.torch.load_file(os.path.join(input_dir, self.prediction_model_name,
                                                                          "diffusion_pytorch_model.safetensors"))
                    model.load_state_dict(state_dict)
                    del state_dict
                elif isinstance(unwrapped_model, (UNet2DConditionModel, UNet2DConditionModelGated)):
                    load_model = UNet2DConditionModelGated.from_pretrained(input_dir,
                                                                           subfolder=self.prediction_model_name)
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    model.total_macs = load_model.total_macs
                    del load_model
                elif isinstance(unwrapped_model, GatedFluxTransformer2DModel):
                    load_model = GatedFluxTransformer2DModel.from_pretrained(input_dir,
                                                                             subfolder=self.prediction_model_name)
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    model.total_macs = load_model.total_macs
                    del load_model
                elif isinstance(model, HyperStructure):
                    load_model = HyperStructure.from_pretrained(input_dir, subfolder="hypernet")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, StructureVectorQuantizer):
                    load_model = StructureVectorQuantizer.from_pretrained(input_dir, subfolder="quantizer")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def init_trackers(self, resume=False):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initialize automatically on the main process.
        if self.accelerator.is_main_process:
            def cfg2dict(cfg: DictConfig) -> Dict:
                """
                Recursively convert OmegaConf to vanilla dict
                :param cfg:
                :return:
                """
                cfg_dict = {}
                for k, v in cfg.items():
                    if type(v) == DictConfig:
                        cfg_dict[k] = cfg2dict(v)
                    else:
                        cfg_dict[k] = v
                return cfg_dict

            tracker_config = cfg2dict(self.config)
            if self.config.wandb_run_name is None:
                self.config.wandb_run_name = (
                    f"{self.config.data.dataset_name if self.config.data.dataset_name else self.config.data.data_dir.split('/')[-1]}-"
                    f"{self.config.data.max_train_samples}-steps:{self.config.training.max_train_steps}-"
                    f"h_lr:{self.config.training.optim.hypernet_learning_rate}-"
                    f"q_lr:{self.config.training.optim.quantizer_learning_rate}")
            self.accelerator.init_trackers(self.config.tracker_project_name, tracker_config,
                                           init_kwargs={"wandb": {"name": self.config.wandb_run_name,
                                                                  "dir": self.config.training.logging.wandb_log_dir,
                                                                  "resume": resume}})

    def configure_logging(self):
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            set_verbosity_warning()
            transformers.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            set_verbosity_error()
            transformers.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def init_prompts(self):
        if os.path.isfile(self.config.data.prompts[0]):
            with open(self.config.data.prompts[0], "r") as f:
                self.config.data.prompts = [line.strip() for line in f.readlines()]
        elif os.path.isdir(self.config.data.prompts[0]):
            validation_prompts_dir = self.config.data.prompts[0]
            prompts = []
            for d in validation_prompts_dir:
                files = [os.path.join(d, caption_file) for caption_file in os.listdir(d) if
                         caption_file.endswith(".txt")]
                for f in files:
                    with open(f, "r") as f:
                        prompts.extend([line.strip() for line in f.readlines()])

            self.config.data.prompts = prompts

        if self.config.data.max_generated_samples is not None:
            self.config.data.prompts = self.config.data.prompts[
                                       :self.config.data.max_generated_samples]

    def init_lr_scheduler(self):
        lr_scheduler = get_scheduler(
            self.config.training.optim.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.optim.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.training.max_train_steps * self.accelerator.num_processes,
        )
        return lr_scheduler

    def update_config_params(self):
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps)
        if self.config.training.max_train_steps is None:
            self.config.training.max_train_steps = self.config.training.num_train_epochs * self.num_update_steps_per_epoch
            self.overrode_max_train_steps = True

    def save_checkpoint(self, logging_dir, global_step):
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if self.config.training.logging.checkpoints_total_limit is not None:
            checkpoints = os.listdir(logging_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_
            # `checkpoints_total_limit - 1` checkpoints
            if len(checkpoints) >= self.config.training.logging.checkpoints_total_limit:
                num_to_remove = len(
                    checkpoints) - self.config.training.logging.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(logging_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(logging_dir, f"checkpoint-{global_step}")
        self.accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    def load_checkpoint(self):
        first_epoch = 0
        logging_dir = self.config.training.logging.logging_dir
        # Potentially load in the weights and states from a previous save
        if self.config.training.logging.resume_from_checkpoint:
            if self.config.training.logging.resume_from_checkpoint != "latest":
                path = self.config.training.logging.resume_from_checkpoint
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(logging_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.config.training.logging.resume_from_checkpoint}' "
                    f"does not exist. Starting a new training run."
                )
                self.config.training.logging.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                if self.config.training.logging.resume_from_checkpoint != "latest":
                    self.accelerator.load_state(path)
                else:
                    self.accelerator.load_state(os.path.join(logging_dir, path))
                global_step = int(os.path.basename(path).split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // self.num_update_steps_per_epoch

        else:
            initial_global_step = 0

        return initial_global_step, first_epoch

    def init_weight_dtype(self):
        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder
        # and non-lora unet,transformer)
        # to half-precision as these weights are only used for inference, keeping weights in full precision is not
        # required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            self.config.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            self.config.mixed_precision = self.accelerator.mixed_precision

    def update_train_steps(self):
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.config.training.max_train_steps = self.config.training.num_train_epochs * num_update_steps_per_epoch
        # Afterward we recalculate our number of training epochs
        self.config.training.num_train_epochs = math.ceil(
            self.config.training.max_train_steps / num_update_steps_per_epoch)

    def create_logging_dir(self):
        logging_dir = self.config.training.logging.logging_dir
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.config.training.logging.logging_dir is not None:
                os.makedirs(self.config.training.logging.logging_dir, exist_ok=True)

                # dump the args to a yaml file
                logger.info("Project config")
                logger.info(OmegaConf.to_yaml(self.config))
                OmegaConf.save(self.config, os.path.join(self.config.training.logging.logging_dir, "config.yaml"))

            if self.config.training.hf_hub.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=self.config.training.hf_hub.hub_model_id or Path(logging_dir).name, exist_ok=True,
                    token=self.config.training.hf_hub.hub_token
                ).repo_id

    def cast_block_act_hooks(self, prediction_model, dicts):
        def get_activation(activation, name, residuals_present):
            if residuals_present:
                def hook(model, input, output):
                    activation[name] = output[0]
            else:
                def hook(model, input, output):
                    activation[name] = output
            return hook

        prediction_model = self.accelerator.unwrap_model(prediction_model)
        for i in range(len(prediction_model.down_blocks)):
            prediction_model.down_blocks[i].register_forward_hook(get_activation(dicts, 'd' + str(i), True))
        prediction_model.mid_block.register_forward_hook(get_activation(dicts, 'm', False))
        for i in range(len(prediction_model.up_blocks)):
            prediction_model.up_blocks[i].register_forward_hook(get_activation(dicts, 'u' + str(i), False))

    def save_model_card(
            self,
            repo_id: str,
            images=None,
            repo_folder=None,
    ):
        img_str = ""
        if len(images) > 0:
            image_grid = make_image_grid(images, 1, len(self.config.data.prompts))
            image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
            img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

        yaml = f"""
                ---
                license: creativeml-openrail-m
                base_model: {self.config.pretrained_model_name_or_path}
                datasets:
                - {self.config.data.dataset_name}
                tags:
                - stable-diffusion
                - stable-diffusion-diffusers
                - text-to-image
                - diffusers
                inference: true
                ---
                    """
        model_card = f"""
                # Text-to-image finetuning - {repo_id}

                This pipeline was pruned from **{self.config.pretrained_model_name_or_path}**
                 on the **{self.config.data.dataset_name}** dataset.
                  Below are some example images generated with the finetuned pipeline using the following prompts: 
                  {self.config.data.prompts}: \n
                {img_str}

                ## Pipeline usage

                You can use the pipeline like so:

                ```python
                from diffusers import DiffusionPipeline
                import torch

                pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
                prompt = "{self.config.data.prompts[0]}"
                image = pipeline(prompt).images[0]
                image.save("my_image.png")
                ```

                ## Training info

                These are the key hyperparameters used during training:

                * Epochs: {self.config.training.num_train_epochs}
                * Hypernet Learning rate: {self.config.training.optim.hypernet_learning_rate}
                * Quantizer Learning rate: {self.config.training.optim.quantizer_learning_rate}
                * Batch size: {self.config.data.dataloader.train_batch_size}
                * Gradient accumulation steps: {self.config.training.gradient_accumulation_steps}
                * Image resolution: {self.config.model.prediction_model.resolution}
                * Mixed-precision: {self.config.mixed_precision}

                """
        wandb_info = ""
        if is_wandb_available():
            wandb_run_url = None
            if wandb.run is not None:
                wandb_run_url = wandb.run.url

        if self.config.wandb_run_url is not None:
            wandb_info = f"""
                More information on all the CLI arguments and the environment are available on your
                 [`wandb` run page]({self.config.wandb_run_url}).
                """

        model_card += wandb_info

        with open(os.path.join(repo_folder, "../README.md"), "w") as f:
            f.write(yaml + model_card)

    def get_pipeline(self):
        self.init_weight_dtype()
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if len(self.accelerator.trackers) == 0:
            self.init_trackers()

        if self.hyper_net:
            self.hyper_net.eval()
        if self.quantizer:
            self.quantizer.eval()
        pipeline = StableDiffusionPruningPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.accelerator.unwrap_model(self.vae),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            unet=self.accelerator.unwrap_model(self.prediction_model),
            safety_checker=None,
            revision=self.config.revision,
            torch_dtype=self.weight_dtype,
            hyper_net=self.accelerator.unwrap_model(self.hyper_net),
            quantizer=self.accelerator.unwrap_model(self.quantizer),
        )
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=not self.accelerator.is_main_process)

        if self.config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

    @torch.no_grad()
    def depth_analysis(self, n_consecutive_blocks=1):
        logger.info("Generating depth analysis samples from the given prompts... ")
        pipeline = self.get_pipeline()
        prediction_model_unwrapped = self.unwrap_model(self.prediction_model)

        image_output_dir = os.path.join(self.config.training.logging.logging_dir, "depth_analysis_images")
        os.makedirs(image_output_dir, exist_ok=True)

        n_depth_pruned_blocks = sum([sum(d) for d in prediction_model_unwrapped.get_structure()['depth']])

        # index n_depth_pruned_blocks is for no pruning
        images = {i: [] for i in range(n_depth_pruned_blocks + 1)}

        for d_block in range(n_depth_pruned_blocks + 1):
            logger.info(f"Generating samples for depth block {d_block}...")
            progress_bar = tqdm(
                range(0, len(self.config.data.prompts)),
                initial=0,
                desc="Depth Analysis Steps",
                # Only show the progress bar once on each machine.
                disable=not self.accelerator.is_main_process,
            )
            for step in range(0, len(self.config.data.prompts),
                              self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes):
                batch = self.config.data.prompts[
                        step:step + self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes]
                with self.accelerator.split_between_processes(batch) as batch:
                    if self.config.seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)

                    if d_block == n_depth_pruned_blocks:
                        gen_images = pipeline.depth_analysis(batch,
                                                             num_inference_steps=self.config.training.num_inference_steps,
                                                             generator=generator, depth_index=None,
                                                             output_type="pt").images
                    else:
                        if n_consecutive_blocks > 1:
                            d_blocks = [(d_block + i) % n_depth_pruned_blocks for i in range(n_consecutive_blocks)]
                        gen_images = pipeline.depth_analysis(batch,
                                                             num_inference_steps=self.config.training.num_inference_steps,
                                                             generator=generator, depth_index=d_blocks,
                                                             output_type="pt").images

                    gen_images = self.accelerator.gather(gen_images)

                    # append gen_images to images dict at the same key
                    images[d_block] += gen_images

                progress_bar.update(self.config.data.dataloader.validation_batch_size * self.accelerator.num_processes)

        image_grids = {}
        for i, image in images.items():
            # convert image from tensor to PIL image
            image = [torchvision.transforms.ToPILImage()(img) for img in image]

            # make an image grid with 4 columns
            image_grid = make_image_grid(image[:4 * (len(images) // 4)], 4, len(images) // 4)
            image_grid.save(os.path.join(image_output_dir, f"depth_{i}.png"))
            image_grids[i] = image_grid

        self.accelerator.log(
            {"depth analysis": [wandb.Image(image_grid, caption=f"Depth: {i}") for i, image_grid in
                                image_grids.items()]}
        )
        return gen_images

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


class UnetPruner(Trainer):
    def __init__(self, config: DictConfig):
        self.prediction_model_name = "unet"
        super().__init__(config)

    def init_models(self):

        logger.info("Loading models...")
        noise_scheduler = DDIMScheduler.from_pretrained(self.config.pretrained_model_name_or_path,
                                                        subfolder="scheduler")

        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.config.revision
        )

        # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
        # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
        # will try to assign the same optimizer with the same weights to all models during
        # `deepspeed.initialize`, which of course doesn't work.
        #
        # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
        # frozen models from being partitioned during `zero.Init` which gets called during
        # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
        # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.config.revision
            )
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="vae", revision=self.config.revision
            )

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        mpnet_tokenizer = AutoTokenizer.from_pretrained(self.config.prompt_encoder_model_name_or_path)
        mpnet_model = AutoModel.from_pretrained(self.config.prompt_encoder_model_name_or_path)

        unet = UNet2DConditionModelGated.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.prediction_model_name,
            revision=self.config.non_ema_revision,
            down_block_types=tuple(self.config.model.prediction_model.unet_down_blocks),
            mid_block_type=self.config.model.prediction_model.unet_mid_block,
            up_block_types=tuple(self.config.model.prediction_model.unet_up_blocks),
            gated_ff=self.config.model.prediction_model.gated_ff,
            ff_gate_width=self.config.model.prediction_model.ff_gate_width,

        )

        unet.freeze()

        unet_structure = unet.get_structure()

        hyper_net = HyperStructure(input_dim=mpnet_model.config.hidden_size,
                                   structure=unet_structure,
                                   wn_flag=self.config.model.hypernet.weight_norm,
                                   linear_bias=self.config.model.hypernet.linear_bias,
                                   single_arch_param=self.config.model.hypernet.single_arch_param
                                   )

        quantizer = StructureVectorQuantizer(n_e=self.config.model.quantizer.num_arch_vq_codebook_embeddings,
                                             structure=unet_structure,
                                             beta=self.config.model.quantizer.arch_vq_beta,
                                             temperature=self.config.model.quantizer.quantizer_T,
                                             base=self.config.model.quantizer.quantizer_base,
                                             depth_order=list(self.config.model.quantizer.depth_order),
                                             non_zero_width=self.config.model.quantizer.non_zero_width,
                                             resource_aware_normalization=self.config.model.quantizer.resource_aware_normalization,
                                             optimal_transport=self.config.model.quantizer.optimal_transport
                                             )
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.mpnet_tokenizer = mpnet_tokenizer
        self.mpnet_model = mpnet_model
        self.prediction_model = unet
        self.hyper_net = hyper_net
        self.quantizer = quantizer

    def init_datasets(self):
        logger.info("Loading datasets...")
        dataset = get_dataset(self.config.data)
        return dataset

    def prepare_with_accelerator(self):
        # Prepare everything with our `accelerator`.
        (self.prediction_model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.prompt_dataloader,
         self.quantizer_embeddings_dataloader, self.lr_scheduler, self.hyper_net,
         self.quantizer) = (self.accelerator.prepare(self.prediction_model, self.optimizer, self.train_dataloader,
                                                     self.eval_dataloader, self.prompt_dataloader,
                                                     self.quantizer_embeddings_dataloader, self.lr_scheduler,
                                                     self.hyper_net,
                                                     self.quantizer
                                                     ))

    def init_dataloaders(self, data_collate_fn, prompts_collate_fn):
        train_dataloader, eval_dataloader, prompt_dataloader = self.get_main_dataloaders(data_collate_fn,
                                                                                         prompts_collate_fn)
        n_e = self.quantizer.n_e
        # used for generating unconditional samples from experts. Can be removed if not needed.
        q_embedding_dataloader = torch.utils.data.DataLoader(torch.arange(n_e),
                                                             batch_size=self.config.data.dataloader.image_generation_batch_size * self.accelerator.num_processes,
                                                             shuffle=False,
                                                             num_workers=self.config.data.dataloader.dataloader_num_workers)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.prompt_dataloader = prompt_dataloader
        self.quantizer_embeddings_dataloader = q_embedding_dataloader

    def init_optimizer(self):
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        scaling_factor = (self.config.training.gradient_accumulation_steps *
                          self.config.data.dataloader.train_batch_size * self.accelerator.num_processes)

        if self.config.training.optim.scale_lr:
            self.config.training.optim.hypernet_learning_rate = (
                    self.config.training.optim.hypernet_learning_rate * math.sqrt(scaling_factor)
            )
            self.config.training.optim.quantizer_learning_rate = (
                    self.config.training.optim.quantizer_learning_rate * math.sqrt(scaling_factor)
            )
            self.config.training.optim.prediction_model_learning_rate = (
                    self.config.training.optim.prediction_model_learning_rate * math.sqrt(scaling_factor)
            )

        params = [
            {"params": self.hyper_net.parameters(),
             "lr": self.config.training.optim.hypernet_learning_rate,
             "weight_decay": self.config.training.optim.hypernet_weight_decay},
            {"params": self.quantizer.parameters(),
             "lr": self.config.training.optim.quantizer_learning_rate,
             "weight_decay": self.config.training.optim.quantizer_weight_decay},
            {"params": [p for p in self.prediction_model.parameters() if p.requires_grad],
             "lr": self.config.training.optim.prediction_model_learning_rate,
             "weight_decay": self.config.training.optim.prediction_model_weight_decay},
        ]

        optimizer = self.get_optimizer(params)
        return optimizer

    def init_losses(self):
        resource_loss = ResourceLoss(p=self.config.training.losses.resource_loss.pruning_target,
                                     loss_type=self.config.training.losses.resource_loss.type)

        contrastive_loss = ContrastiveLoss(
            arch_vector_temperature=self.config.training.losses.contrastive_loss.arch_vector_temperature,
            prompt_embedding_temperature=self.config.training.losses.contrastive_loss.prompt_embedding_temperature)

        ddpm_loss = F.mse_loss
        distillation_loss = F.mse_loss

        self.ddpm_loss = ddpm_loss
        self.distillation_loss = distillation_loss
        self.resource_loss = resource_loss
        self.contrastive_loss = contrastive_loss

    def train(self):
        self.init_weight_dtype()

        # Move text_encoder and vae to gpu and cast to weight_dtype

        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        if hasattr(self, "text_encoder2"):
            self.text_encoder2.to(self.accelerator.device, dtype=self.weight_dtype)

        self.update_train_steps()
        # Train!
        logging_dir = self.config.training.logging.logging_dir
        total_batch_size = (self.config.data.dataloader.train_batch_size * self.accelerator.num_processes *
                            self.config.training.gradient_accumulation_steps)

        initial_global_step, first_epoch = self.load_checkpoint()
        global_step = initial_global_step

        logger.info("***** Running pruning *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.config.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.data.dataloader.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.training.max_train_steps}")

        progress_bar = tqdm(
            range(0, self.config.training.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )

        self.block_activations = {}
        self.cast_block_act_hooks(self.prediction_model, self.block_activations)

        for epoch in range(first_epoch, self.config.training.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.prediction_model):
                    if "pixel_values" in batch and batch["pixel_values"].numel() == 0:
                        continue

                    train_loss = 0.0
                    self.hyper_net.train()
                    self.quantizer.train()
                    self.prediction_model.eval()

                    # Calculating the MACs of each module of the model in the first iteration.
                    if global_step == initial_global_step:
                        self.count_macs(batch)

                    # pruning target is for total macs. we calculate loss for prunable macs.
                    if global_step == 0:
                        self.update_pruning_target()

                    pretrain = (self.config.training.hypernet_pretraining_steps and
                                global_step < self.config.training.hypernet_pretraining_steps)
                    (loss, diff_loss, distillation_loss, block_loss, contrastive_loss, resource_loss,
                     arch_vectors_similarity, resource_ratio, macs_dict, arch_vector_quantized,
                     quantizer_embedding_pairwise_similarity, batch_resource_ratios) = self.step(batch,
                                                                                                 pretrain=pretrain)

                    avg_loss = loss
                    train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

                    # Back-propagate
                    try:
                        self.accelerator.backward(loss)
                    except RuntimeError as e:
                        if "returned nan values" in str(e):
                            logger.error("NaNs detected in the loss. Skipping batch.")
                            self.optimizer.zero_grad()
                            continue
                        else:
                            raise e

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        log_dict = {
                            "training/loss": train_loss,
                            "training/diffusion_loss": diff_loss,
                            "training/distillation_loss": distillation_loss.detach().item(),
                            "training/block_loss": block_loss.detach().item(),
                            "training/resource_loss": resource_loss.detach().item(),
                            "training/contrastive_loss": contrastive_loss.detach().item(),
                            "training/hyper_net_lr": self.lr_scheduler.get_last_lr()[0],
                            "training/quantizer_lr": self.lr_scheduler.get_last_lr()[1],
                            "training/resource_ratio": resource_ratio.detach().item(),
                        }
                        for k, v in macs_dict.items():
                            if isinstance(v, torch.Tensor):
                                log_dict[f"training/{k}"] = v.detach().mean().item()
                            else:
                                log_dict[f"training/{k}"] = v

                        self.accelerator.log(log_dict)

                        logs = {"diff_loss": diff_loss.detach().item(),
                                "dist_loss": distillation_loss.detach().item(),
                                "block_loss": block_loss.detach().item(),
                                "c_loss": contrastive_loss.detach().item(),
                                "r_loss": resource_loss.detach().item(),
                                "step_loss": loss.detach().item(),
                                "h_lr": self.lr_scheduler.get_last_lr()[0],
                                "q_lr": self.lr_scheduler.get_last_lr()[1],
                                }
                        progress_bar.set_postfix(**logs)

                        if global_step % self.config.training.validation_steps == 0:
                            if self.eval_dataset is not None:
                                self.validate(pretrain=pretrain)

                        if (global_step % self.config.training.image_logging_steps == 0 or
                                (epoch == self.config.training.num_train_epochs - 1 and step == len(
                                    self.train_dataloader) - 1)):
                            img_log_dict = {
                                "images/arch vector pairwise similarity image": wandb.Image(
                                    arch_vectors_similarity),
                                "images/quantizer_embedding_pairwise_similarity": wandb.Image(
                                    quantizer_embedding_pairwise_similarity)
                            }

                            with torch.no_grad():
                                batch_resource_ratios = self.accelerator.gather_for_metrics(
                                    batch_resource_ratios).cpu().numpy()

                            # make sure the number of rows is a multiple of 16 by adding zeros. This can be avoided by
                            # removing corrupt images from the dataset.
                            if batch_resource_ratios.shape[0] % 16 != 0:
                                batch_resource_ratios = np.concatenate(
                                    [batch_resource_ratios, np.zeros((16 - len(batch_resource_ratios) % 16, 1))],
                                    axis=0)

                            img_log_dict["images/batch resource ratio heatmap"] = wandb.Image(
                                create_heatmap(batch_resource_ratios, n_rows=16,
                                               n_cols=len(batch_resource_ratios) // 16))
                            self.accelerator.log(img_log_dict, log_kwargs={"wandb": {"commit": False}})

                            # generate some validation images
                            if self.config.data.prompts is not None:
                                val_images = self.generate_samples_from_prompts(pretrain=pretrain)

                            # visualize the quantizer embeddings
                            self.log_quantizer_embedding_samples(global_step)

                        global_step += 1

                    if global_step >= self.config.training.max_train_steps:
                        break

            # checkpoint at the end of each epoch
            if self.accelerator.is_main_process:
                self.save_checkpoint(logging_dir, global_step)

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.config.push_to_hub:
                self.save_model_card(self.repo_id, val_images, repo_folder=self.config.output_dir)
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=logging_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self.accelerator.end_training()

    @torch.no_grad()
    def validate(self, pretrain=False):
        self.init_weight_dtype()
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if len(self.accelerator.trackers) == 0:
            self.init_trackers()

        progress_bar = tqdm(
            range(0, len(self.eval_dataloader)),
            initial=0,
            desc="Val Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_main_process,
        )

        self.hyper_net.eval()
        self.quantizer.eval()
        self.prediction_model.eval()
        (total_val_loss, total_diff_loss, total_distillation_loss, total_block_loss, total_c_loss,
         total_r_loss) = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for step, batch in enumerate(self.eval_dataloader):
            if batch["pixel_values"].numel() == 0:
                continue
            (loss, diff_loss, distillation_loss, block_loss, contrastive_loss, resource_loss,
             _, _, _, _, _, _) = self.step(batch, pretrain=pretrain)
            # Gather the losses across all processes for logging (if we use distributed training).
            total_val_loss += loss.item()
            total_diff_loss += diff_loss.item()
            total_distillation_loss += distillation_loss.item()
            total_block_loss += block_loss.item()
            total_c_loss += contrastive_loss.item()
            total_r_loss += resource_loss.item()
            progress_bar.update(1)

        total_val_loss /= len(self.eval_dataloader)
        total_diff_loss /= len(self.eval_dataloader)
        total_c_loss /= len(self.eval_dataloader)
        total_distillation_loss /= len(self.eval_dataloader)
        total_block_loss /= len(self.eval_dataloader)
        total_r_loss /= len(self.eval_dataloader)

        total_val_loss = self.accelerator.reduce(torch.tensor(total_val_loss, device=self.accelerator.device),
                                                 "mean").item()
        total_diff_loss = self.accelerator.reduce(torch.tensor(total_diff_loss, device=self.accelerator.device),
                                                  "mean").item()
        total_c_loss = self.accelerator.reduce(torch.tensor(total_c_loss, device=self.accelerator.device),
                                               "mean").item()
        total_r_loss = self.accelerator.reduce(torch.tensor(total_r_loss, device=self.accelerator.device),
                                               "mean").item()
        total_distillation_loss = self.accelerator.reduce(
            torch.tensor(total_distillation_loss, device=self.accelerator.device), "mean").item()
        total_block_loss = self.accelerator.reduce(torch.tensor(total_block_loss, device=self.accelerator.device),
                                                   "mean").item()

        self.accelerator.log({
            "validation/loss": total_val_loss,
            "validation/diffusion_loss": total_diff_loss,
            "validation/distillation_loss": total_distillation_loss,
            "validation/block_loss": total_block_loss,
            "validation/resource_loss": total_r_loss,
            "validation/contrastive_loss": total_c_loss,
        },
            log_kwargs={"wandb": {"commit": False}})

    def step(self, batch, pretrain=False):
        prediction_model_unwrapped = self.unwrap_model(self.prediction_model)
        hyper_net_unwrapped = self.unwrap_model(self.hyper_net)
        quantizer_unwrapped = self.unwrap_model(self.quantizer)

        latents = self.vae.encode(batch["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.config.model.prediction_model.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.config.model.prediction_model.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if self.config.model.prediction_model.input_perturbation:
            new_noise = noise + self.config.model.prediction_model.input_perturbation * torch.randn_like(noise)

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        if self.config.model.prediction_model.max_scheduler_steps is None:
            self.config.model.prediction_model.max_scheduler_steps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, self.config.model.prediction_model.max_scheduler_steps, (bsz,),
                                  device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward process)
        if self.config.model.prediction_model.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = batch["prompt_embeds"].to(self.accelerator.device, dtype=self.weight_dtype)
        text_embeddings = batch["mpnet_embeddings"]

        arch_vector = self.hyper_net(text_embeddings)
        arch_vector_quantized, _ = self.quantizer(arch_vector)

        arch_vector = quantizer_unwrapped.gumbel_sigmoid_trick(arch_vector)

        if hyper_net_unwrapped.single_arch_param:
            arch_vector = arch_vector.repeat(text_embeddings.shape[0], 1)
            hyper_net_unwrapped.arch_gs = arch_vector

        arch_vector_width_depth_normalized = quantizer_unwrapped.width_depth_normalize(arch_vector)
        with torch.no_grad():
            quantizer_embeddings = quantizer_unwrapped.get_codebook_entry_gumbel_sigmoid(
                torch.arange(quantizer_unwrapped.n_e, device=self.accelerator.device),
                hard=True).detach()

            quantizer_embeddings /= quantizer_embeddings.norm(dim=-1, keepdim=True)
            quantizer_embeddings_pairwise_similarity = quantizer_embeddings @ quantizer_embeddings.t()

            text_embeddings_list = [torch.zeros_like(text_embeddings) for _ in
                                    range(self.accelerator.num_processes)]
            arch_vector_list = [torch.zeros_like(arch_vector) for _ in
                                range(self.accelerator.num_processes)]

            if self.accelerator.num_processes > 1:
                torch.distributed.all_gather(text_embeddings_list, text_embeddings)
                torch.distributed.all_gather(arch_vector_list, arch_vector_width_depth_normalized)
            else:
                text_embeddings_list[self.accelerator.process_index] = text_embeddings
                arch_vector_list[self.accelerator.process_index] = arch_vector_width_depth_normalized

        text_embeddings_list[self.accelerator.process_index] = text_embeddings
        arch_vector_list[self.accelerator.process_index] = arch_vector_width_depth_normalized
        text_embeddings_list = torch.cat(text_embeddings_list, dim=0)
        arch_vector_list = torch.cat(arch_vector_list, dim=0)

        # During hyper_net pretraining, we don't cluster the architecture vector and directly use it.
        if pretrain:
            arch_vectors_separated = hyper_net_unwrapped.transform_structure_vector(arch_vector)
        else:
            arch_vectors_separated = hyper_net_unwrapped.transform_structure_vector(arch_vector_quantized)

        contrastive_loss, arch_vectors_similarity = self.contrastive_loss(text_embeddings_list, arch_vector_list,
                                                                          return_similarity=True)

        # Get the target for loss depending on the prediction type
        if self.config.model.prediction_model.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.config.model.prediction_model.prediction_type)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        with torch.no_grad():
            full_arch_vector = torch.ones_like(arch_vector)
            full_arch_vector = hyper_net_unwrapped.transform_structure_vector(full_arch_vector)
            prediction_model_unwrapped.set_structure(full_arch_vector)
            full_model_pred = self.prediction_model(noisy_latents, timesteps, encoder_hidden_states).sample.detach()
            teacher_block_activations = self.block_activations.copy()

        prediction_model_unwrapped.set_structure(arch_vectors_separated)
        # Predict the noise residual and compute loss
        model_pred = self.prediction_model(noisy_latents, timesteps, encoder_hidden_states).sample
        student_block_activations = self.block_activations.copy()

        if self.config.training.losses.diffusion_loss.snr_gamma is None:
            loss = self.ddpm_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                    torch.stack(
                        [snr,
                         self.config.training.losses.diffusion_loss.snr_gamma * torch.ones_like(timesteps)],
                        dim=1).min(dim=1)[0] / snr
            )

            loss = self.ddpm_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        distillation_loss = self.distillation_loss(model_pred.float(), full_model_pred.float(), reduction="mean")

        block_loss = torch.tensor(0.0, device=self.accelerator.device)
        for key in student_block_activations.keys():
            block_loss += self.distillation_loss(student_block_activations[key],
                                                 teacher_block_activations[key].detach(),
                                                 reduction="mean")
        block_loss /= len(student_block_activations)

        macs_dict = prediction_model_unwrapped.calc_macs()
        curr_macs = macs_dict['cur_prunable_macs']

        # The reason is that sanity['prunable_macs'] does not have depth-related pruning
        # macs like skip connections of resnets in it.
        resource_ratios = (curr_macs / (
            prediction_model_unwrapped.resource_info_dict['cur_prunable_macs'].squeeze()))

        resource_loss = self.resource_loss(resource_ratios.mean())

        max_loss = 1. - torch.max(resource_ratios)
        std_loss = -torch.std(resource_ratios)
        with torch.no_grad():
            batch_resource_ratios = macs_dict['cur_prunable_macs'] / (
                prediction_model_unwrapped.resource_info_dict['cur_prunable_macs'].squeeze())

        diff_loss = loss.clone().detach().mean()
        loss += self.config.training.losses.resource_loss.weight * resource_loss
        loss += self.config.training.losses.contrastive_loss.weight * contrastive_loss
        loss += self.config.training.losses.distillation_loss.weight * distillation_loss
        loss += self.config.training.losses.block_loss.weight * block_loss
        loss += self.config.training.losses.std_loss.weight * std_loss
        loss += self.config.training.losses.max_loss.weight * max_loss

        return (
            loss, diff_loss, distillation_loss, block_loss, contrastive_loss, resource_loss,
            arch_vectors_similarity, resource_ratios.mean(),
            macs_dict, arch_vector_quantized, quantizer_embeddings_pairwise_similarity, batch_resource_ratios)

    @torch.no_grad()
    def count_macs(self, batch):
        hyper_net_unwrapped = self.unwrap_model(self.hyper_net)
        quantizer_unwrapped = self.unwrap_model(self.quantizer)
        prediction_model_unwrapped = self.unwrap_model(self.prediction_model)

        arch_vecs_separated = hyper_net_unwrapped.transform_structure_vector(
            torch.ones((1, quantizer_unwrapped.vq_embed_dim), device=self.accelerator.device))

        prediction_model_unwrapped.set_structure(arch_vecs_separated)

        latents = self.vae.encode(batch["pixel_values"][:1].to(self.weight_dtype)).latent_dist.sample()
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,),
                                  device=self.accelerator.device).long()
        encoder_hidden_states = batch["prompt_embeds"][:1].to(self.accelerator.device, dtype=self.weight_dtype)

        macs, params = count_ops_and_params(self.prediction_model,
                                            {'sample': latents,
                                             'timestep': timesteps,
                                             'encoder_hidden_states': encoder_hidden_states})

        logger.info(
            "{}'s Params/MACs calculated by OpCounter:\tparams: {:.3f}M\t MACs: {:.3f}G".format(
                self.prediction_model_name, params / 1e6, macs / 1e9))

        sanity_macs_dict = prediction_model_unwrapped.calc_macs()
        prunable_macs_list = [[e / sanity_macs_dict['prunable_macs'] for e in elem] for elem in
                              prediction_model_unwrapped.get_prunable_macs()]

        prediction_model_unwrapped.prunable_macs_list = prunable_macs_list
        prediction_model_unwrapped.resource_info_dict = sanity_macs_dict

        quantizer_unwrapped.set_prunable_macs_template(prunable_macs_list)

        sanity_string = "Our MACs calculation:\t"
        for k, v in sanity_macs_dict.items():
            if isinstance(v, torch.Tensor):
                sanity_string += f" {k}: {v.item() / 1e9:.3f}\t"
            else:
                sanity_string += f" {k}: {v / 1e9:.3f}\t"
        logger.info(sanity_string)

    @torch.no_grad()
    def update_pruning_target(self):
        prediction_model_unwrapped = self.unwrap_model(self.prediction_model)
        p = self.config.training.losses.resource_loss.pruning_target

        p_actual = (1 - (1 - p) * prediction_model_unwrapped.resource_info_dict['total_macs'] /
                    prediction_model_unwrapped.resource_info_dict['cur_prunable_macs']).item()

        self.resource_loss.p = p_actual

    @torch.no_grad()
    def generate_samples_from_prompts(self, pretrain=False):
        logger.info("Generating samples from the given prompts... ")

        pipeline = self.get_pipeline()

        images = []
        prompts_resource_ratios = []

        for step, batch in enumerate(self.prompt_dataloader):
            if self.config.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
            gen_images, _, resource_ratios = pipeline(batch["prompts"],
                                                      num_inference_steps=self.config.training.num_inference_steps,
                                                      generator=generator, output_type="pt",
                                                      return_mapped_indices=True,
                                                      hyper_net_input=batch["mpnet_embeddings"],
                                                      pretrain=pretrain
                                                      )
            gen_images = gen_images.images
            gen_images = self.accelerator.gather_for_metrics(gen_images)
            resource_ratios = self.accelerator.gather_for_metrics(resource_ratios)
            images += gen_images
            prompts_resource_ratios += resource_ratios

        images = [torchvision.transforms.ToPILImage()(img) for img in images]

        imgs_len = len(images)
        n_cols = 4 if imgs_len % 4 == 0 else 3 if imgs_len % 3 == 0 else 2 if imgs_len % 2 == 0 else 1
        prompts_resource_ratios = torch.cat(prompts_resource_ratios, dim=0).cpu().numpy()
        prompts_resource_ratios_images = create_heatmap(prompts_resource_ratios, n_rows=n_cols,
                                                        n_cols=len(prompts_resource_ratios) // n_cols)
        images = make_image_grid(images, n_cols, len(images) // n_cols)

        self.accelerator.log(
            {
                "images/prompt images": wandb.Image(images),
                "images/prompts resource ratio heatmap": wandb.Image(prompts_resource_ratios_images),
            },
            log_kwargs={"wandb": {"commit": False}}
        )
        return images

    @torch.no_grad()
    def log_quantizer_embedding_samples(self, step, save_to_disk=False):
        logger.info("Sampling from quantizer... ")

        pipeline = self.get_pipeline()

        images = []
        quantizer_embedding_gumbel_sigmoid = []
        embeddings_resource_ratios = []

        for step, indices in enumerate(self.quantizer_embeddings_dataloader):
            if self.config.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)

            gen_images, quantizer_embed_gs, resource_ratios = pipeline.quantizer_samples(indices=indices,
                                                                                         num_inference_steps=self.config.training.num_inference_steps,
                                                                                         generator=generator,
                                                                                         output_type="pt")
            gen_images = gen_images.images
            gen_images = self.accelerator.gather_for_metrics(gen_images)
            quantizer_embed_gs = self.accelerator.gather_for_metrics(quantizer_embed_gs)
            resource_ratios = self.accelerator.gather_for_metrics(resource_ratios)
            quantizer_embedding_gumbel_sigmoid += quantizer_embed_gs
            images += gen_images
            embeddings_resource_ratios += resource_ratios

        quantizer_embedding_gumbel_sigmoid = torch.cat(quantizer_embedding_gumbel_sigmoid, dim=0)
        images = [torchvision.transforms.ToPILImage()(img) for img in images]
        imgs_len = len(images)

        n_cols = 4 if imgs_len % 4 == 0 else 3 if imgs_len % 3 == 0 else 2 if imgs_len % 2 == 0 else 1

        embeddings_resource_ratios = torch.cat(embeddings_resource_ratios, dim=0).cpu().numpy()
        embeddings_resource_ratios_images = create_heatmap(embeddings_resource_ratios, n_rows=n_cols,
                                                           n_cols=len(embeddings_resource_ratios) // n_cols)
        images = make_image_grid(images, n_cols, len(images) // n_cols)

        if self.accelerator.is_main_process and save_to_disk:
            image_output_dir = os.path.join(self.config.training.logging.logging_dir, "quantizer_embedding_images",
                                            f"step_{step}")
            os.makedirs(image_output_dir, exist_ok=True)
            torch.save(quantizer_embedding_gumbel_sigmoid, os.path.join(image_output_dir,
                                                                        "quantizer_embeddings_gumbel_sigmoid.pt"))

        self.accelerator.log({"images/quantizer embedding images": wandb.Image(images),
                              "images/embedding resource ratio heatmap": wandb.Image(
                                  embeddings_resource_ratios_images)},
                             log_kwargs={"wandb": {"commit": False}})


class SDXLPruner(UnetPruner):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.tokenizer2 = None
        self.text_encoder2 = None
        self.vae_path = None

    def init_models(self):

        def import_model_class_from_model_name_or_path(
                pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
        ):
            text_encoder_config = PretrainedConfig.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, revision=revision
            )
            model_class = text_encoder_config.architectures[0]

            if model_class == "CLIPTextModel":
                from transformers import CLIPTextModel

                return CLIPTextModel
            elif model_class == "CLIPTextModelWithProjection":
                from transformers import CLIPTextModelWithProjection

                return CLIPTextModelWithProjection
            else:
                raise ValueError(f"{model_class} is not supported.")

        logger.info("Loading models...")
        noise_scheduler = DDIMScheduler.from_pretrained(self.config.pretrained_model_name_or_path,
                                                        subfolder="scheduler")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.config.revision,
            use_fast=False
        )

        tokenizer2 = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=self.config.revision,
            use_fast=False
        )

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.config.pretrained_model_name_or_path, self.config.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.config.pretrained_model_name_or_path, self.config.revision, subfolder="text_encoder_2"
        )

        text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.config.revision,
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=self.config.revision,
        )

        vae_path = (
            self.config.pretrained_model_name_or_path
            if self.config.pretrained_vae_model_name_or_path is None
            else self.config.pretrained_vae_model_name_or_path
        )

        self.vae_path = vae_path

        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if self.config.pretrained_vae_model_name_or_path is None else None,
            revision=self.config.revision,
        )

        # Freeze vae and text encoders.
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

        mpnet_tokenizer = AutoTokenizer.from_pretrained(self.config.prompt_encoder_model_name_or_path)
        mpnet_model = AutoModel.from_pretrained(self.config.prompt_encoder_model_name_or_path)

        unet = UNet2DConditionModelGated.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.prediction_model_name,
            revision=self.config.non_ema_revision,
            down_block_types=tuple(self.config.model.prediction_model.unet_down_blocks),
            mid_block_type=self.config.model.prediction_model.unet_mid_block,
            up_block_types=tuple(self.config.model.prediction_model.unet_up_blocks),
            gated_ff=self.config.model.prediction_model.gated_ff,
            ff_gate_width=self.config.model.prediction_model.ff_gate_width,

        )

        unet.freeze()
        unet_structure = unet.get_structure()

        hyper_net = HyperStructure(input_dim=mpnet_model.config.hidden_size,
                                   structure=unet_structure,
                                   wn_flag=self.config.model.hypernet.weight_norm,
                                   linear_bias=self.config.model.hypernet.linear_bias,
                                   single_arch_param=self.config.model.hypernet.single_arch_param
                                   )

        quantizer = StructureVectorQuantizer(n_e=self.config.model.quantizer.num_arch_vq_codebook_embeddings,
                                             structure=unet_structure,
                                             beta=self.config.model.quantizer.arch_vq_beta,
                                             temperature=self.config.model.quantizer.quantizer_T,
                                             base=self.config.model.quantizer.quantizer_base,
                                             depth_order=list(self.config.model.quantizer.depth_order),
                                             non_zero_width=self.config.model.quantizer.non_zero_width,
                                             resource_aware_normalization=self.config.model.quantizer.resource_aware_normalization,
                                             optimal_transport=self.config.model.quantizer.optimal_transport
                                             )
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.text_encoder = text_encoder_one
        self.text_encoder2 = text_encoder_two
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.mpnet_tokenizer = mpnet_tokenizer
        self.mpnet_model = mpnet_model
        self.prediction_model = unet
        self.hyper_net = hyper_net
        self.quantizer = quantizer


class FluxPruner(UnetPruner):
    def __init__(self, config: DictConfig):
        self.prediction_model_name = "transformer"
        self.tokenizer2 = None
        self.text_encoder2 = None
        self.noise_scheduler_copy = None
        super().__init__(config)

    def init_models(self):
        logger.info("Loading models...")
        self.init_weight_dtype()
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.config.pretrained_model_name_or_path,
                                                                          subfolder="scheduler")
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )

        tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )

        text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )

        vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="vae", revision=self.config.revision,
            torch_dtype=self.weight_dtype,
        )

        # Freeze vae and text encoders.
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        mpnet_tokenizer = AutoTokenizer.from_pretrained(self.config.prompt_encoder_model_name_or_path)
        mpnet_model = AutoModel.from_pretrained(self.config.prompt_encoder_model_name_or_path)

        transformer = GatedFluxTransformer2DModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=self.config.non_ema_revision,
            ff_gate_width=self.config.model.prediction_model.ff_gate_width,
            torch_dtype=self.weight_dtype,
        )

        transformer.freeze()
        transformer_structure = transformer.get_structure()

        hyper_net = HyperStructure(input_dim=mpnet_model.config.hidden_size,
                                   structure=transformer_structure,
                                   wn_flag=self.config.model.hypernet.weight_norm,
                                   linear_bias=self.config.model.hypernet.linear_bias,
                                   single_arch_param=self.config.model.hypernet.single_arch_param,
                                   ).to(self.weight_dtype)

        quantizer = StructureVectorQuantizer(n_e=self.config.model.quantizer.num_arch_vq_codebook_embeddings,
                                             structure=transformer_structure,
                                             beta=self.config.model.quantizer.arch_vq_beta,
                                             temperature=self.config.model.quantizer.quantizer_T,
                                             base=self.config.model.quantizer.quantizer_base,
                                             non_zero_width=self.config.model.quantizer.non_zero_width,
                                             resource_aware_normalization=self.config.model.quantizer.resource_aware_normalization,
                                             optimal_transport=self.config.model.quantizer.optimal_transport,
                                             dtype=self.weight_dtype
                                             ).to(self.weight_dtype)
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder2 = text_encoder_2
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.mpnet_tokenizer = mpnet_tokenizer
        self.mpnet_model = mpnet_model
        self.prediction_model = transformer
        self.hyper_net = hyper_net
        self.quantizer = quantizer

    def init_dataset_preprocessors(self, dataset):

        # Get the column names for input/target.
        column_names = dataset["train"].column_names
        image_column = self.config.data.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{self.config.data.image_column}' needs to be one of: {', '.join(column_names)}"
            )

        caption_column = self.config.data.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{self.config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

        # Preprocessors and transformers
        train_transform, validation_transform = get_transforms(self.config)
        preprocess_train = partial(preprocess_sample,
                                   tokenizers=[self.tokenizer, self.tokenizer2],
                                   text_encoders=[self.text_encoder, self.text_encoder2],
                                   mpnet_model=self.mpnet_model, mpnet_tokenizer=self.mpnet_tokenizer,
                                   transform=train_transform,
                                   image_column=image_column, caption_column=caption_column,
                                   is_train=True,
                                   max_sequence_length=(
                                       self.tokenizer.model_max_length,
                                       self.config.model.tokenizer.max_sequence_length))

        preprocess_validation = partial(preprocess_sample,
                                        tokenizers=[self.tokenizer, self.tokenizer2],
                                        text_encoders=[self.text_encoder, self.text_encoder2],
                                        mpnet_model=self.mpnet_model, mpnet_tokenizer=self.mpnet_tokenizer,
                                        transform=train_transform,
                                        image_column=image_column, caption_column=caption_column,
                                        is_train=False,
                                        max_sequence_length=(
                                            self.tokenizer.model_max_length,
                                            self.config.model.tokenizer.max_sequence_length))

        preprocess_prompts_ = partial(preprocess_prompts, mpnet_model=self.mpnet_model,
                                      mpnet_tokenizer=self.mpnet_tokenizer)

        return preprocess_train, preprocess_validation, preprocess_prompts_

    def cast_block_act_hooks(self, prediction_model, dicts):
        pass

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def step(self, batch, pretrain=False):
        prediction_model_unwrapped = self.unwrap_model(self.prediction_model)
        self.hyper_net = self.hyper_net.to(self.weight_dtype)
        self.quantizer = self.quantizer.to(self.weight_dtype)
        hyper_net_unwrapped = self.unwrap_model(self.hyper_net).to(self.weight_dtype)
        quantizer_unwrapped = self.unwrap_model(self.quantizer).to(self.weight_dtype)

        model_input = self.vae.encode(batch["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        model_input = model_input.to(dtype=self.weight_dtype)

        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels))

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2],
            model_input.shape[3],
            self.accelerator.device,
            self.weight_dtype,
        )

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.config.model.prediction_model.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.config.model.prediction_model.logit_mean,
            logit_std=self.config.model.prediction_model.logit_std,
            mode_scale=self.config.model.prediction_model.mode_scale,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

        # handle guidance
        if self.prediction_model.config.guidance_embeds:
            guidance = torch.tensor([self.config.model.prediction_model.guidance_scale],
                                    device=self.accelerator.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        # Get the text embedding for conditioning
        encoder_hidden_states = batch["prompt_embeds"].to(self.accelerator.device, dtype=self.weight_dtype)
        pooled_projections = batch["pooled_prompt_embeds"].to(self.accelerator.device, dtype=self.weight_dtype)
        text_ids = batch["text_ids"].to(self.accelerator.device)
        text_embeddings = batch["mpnet_embeddings"].to(self.accelerator.device, dtype=self.weight_dtype)

        arch_vector = self.hyper_net(text_embeddings).to(dtype=self.weight_dtype)
        arch_vector_quantized, _ = self.quantizer(arch_vector)
        arch_vector_quantized = arch_vector_quantized.to(dtype=self.weight_dtype)

        arch_vector = quantizer_unwrapped.gumbel_sigmoid_trick(arch_vector).to(dtype=self.weight_dtype)

        if hyper_net_unwrapped.single_arch_param:
            arch_vector = arch_vector.repeat(text_embeddings.shape[0], 1)
            hyper_net_unwrapped.arch_gs = arch_vector

        arch_vector_width_depth_normalized = quantizer_unwrapped.width_depth_normalize(arch_vector)
        with torch.no_grad():
            quantizer_embeddings = quantizer_unwrapped.get_codebook_entry_gumbel_sigmoid(
                torch.arange(quantizer_unwrapped.n_e, device=self.accelerator.device),
                hard=True).detach()

            quantizer_embeddings /= quantizer_embeddings.norm(dim=-1, keepdim=True)
            quantizer_embeddings_pairwise_similarity = quantizer_embeddings @ quantizer_embeddings.t()

            text_embeddings_list = [torch.zeros_like(text_embeddings, dtype=self.weight_dtype) for _ in
                                    range(self.accelerator.num_processes)]
            arch_vector_list = [torch.zeros_like(arch_vector, dtype=self.weight_dtype) for _ in
                                range(self.accelerator.num_processes)]

            if self.accelerator.num_processes > 1:
                torch.distributed.all_gather(text_embeddings_list, text_embeddings)
                torch.distributed.all_gather(arch_vector_list, arch_vector_width_depth_normalized)
            else:
                text_embeddings_list[self.accelerator.process_index] = text_embeddings
                arch_vector_list[self.accelerator.process_index] = arch_vector_width_depth_normalized

        text_embeddings_list[self.accelerator.process_index] = text_embeddings
        arch_vector_list[self.accelerator.process_index] = arch_vector_width_depth_normalized
        text_embeddings_list = torch.cat(text_embeddings_list, dim=0)
        arch_vector_list = torch.cat(arch_vector_list, dim=0).to(dtype=self.weight_dtype)

        # During hyper_net pretraining, we don't cluster the architecture vector and directly use it.
        if pretrain:
            arch_vectors_separated = hyper_net_unwrapped.transform_structure_vector(arch_vector)
        else:
            arch_vectors_separated = hyper_net_unwrapped.transform_structure_vector(arch_vector_quantized)

        contrastive_loss, arch_vectors_similarity = self.contrastive_loss(text_embeddings_list, arch_vector_list,
                                                                          return_similarity=True)

        # with torch.no_grad():
        #     full_arch_vector = torch.ones_like(arch_vector)
        #     full_arch_vector = hyper_net_unwrapped.transform_structure_vector(full_arch_vector)
        #     prediction_model_unwrapped.set_structure(full_arch_vector)
        #     full_model_pred = self.prediction_model(hidden_states=packed_noisy_model_input,
        #                                             timestep=timesteps / 1000,
        #                                             guidance=guidance,
        #                                             pooled_projections=pooled_projections,
        #                                             encoder_hidden_states=encoder_hidden_states,
        #                                             ).sample.detach()
        #     teacher_block_activations = self.block_activations.copy()

        prediction_model_unwrapped.set_structure(arch_vectors_separated)

        # Predict the noise residual
        model_pred = self.prediction_model(
            hidden_states=packed_noisy_model_input,
            # divide it by 1000 for now because we scale it by 1000 in the transformer model
            # (we should not keep it but I want to keep the inputs same for the model for testing)
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=int(model_input.shape[2] * vae_scale_factor / 2),
            width=int(model_input.shape[3] * vae_scale_factor / 2),
            vae_scale_factor=vae_scale_factor,
        )

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.config.model.prediction_model.weighting_scheme,
                                                   sigmas=sigmas)
        # flow matching loss
        target = noise - model_input

        # student_block_activations = self.block_activations.copy()

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        # distillation_loss = self.distillation_loss(model_pred.float(), full_model_pred.float(), reduction="mean")

        # block_loss = torch.tensor(0.0, device=self.accelerator.device)
        # for key in student_block_activations.keys():
        #     block_loss += self.distillation_loss(student_block_activations[key],
        #                                          teacher_block_activations[key].detach(),
        #                                          reduction="mean")
        # block_loss /= len(student_block_activations)

        macs_dict = prediction_model_unwrapped.calc_macs()
        curr_macs = macs_dict['cur_prunable_macs']

        # The reason is that sanity['prunable_macs'] does not have depth-related pruning
        # macs like skip connections of resnets in it.
        resource_ratios = (curr_macs / (
            prediction_model_unwrapped.resource_info_dict['cur_prunable_macs'].squeeze()))

        resource_loss = self.resource_loss(resource_ratios.mean())

        block_loss = torch.tensor(0.0, device=self.accelerator.device)
        distillation_loss = torch.tensor(0.0, device=self.accelerator.device)
        # max_loss = 1. - torch.max(resource_ratios)
        # std_loss = -torch.std(resource_ratios)
        with torch.no_grad():
            batch_resource_ratios = macs_dict['cur_prunable_macs'] / (
                prediction_model_unwrapped.resource_info_dict['cur_prunable_macs'].squeeze())

        diff_loss = loss.clone().detach().mean()
        loss += self.config.training.losses.resource_loss.weight * resource_loss
        loss += self.config.training.losses.contrastive_loss.weight * contrastive_loss
        # loss += self.config.training.losses.distillation_loss.weight * distillation_loss
        # loss += self.config.training.losses.block_loss.weight * block_loss
        # loss += self.config.training.losses.std_loss.weight * std_loss
        # loss += self.config.training.losses.max_loss.weight * max_loss

        return (
            loss, diff_loss, distillation_loss, block_loss, contrastive_loss, resource_loss,
            arch_vectors_similarity, resource_ratios.mean(),
            macs_dict, arch_vector_quantized, quantizer_embeddings_pairwise_similarity, batch_resource_ratios)

    @torch.no_grad()
    def count_macs(self, batch):
        hyper_net_unwrapped = self.unwrap_model(self.hyper_net)
        quantizer_unwrapped = self.unwrap_model(self.quantizer)
        prediction_model_unwrapped = self.unwrap_model(self.prediction_model)

        arch_vecs_separated = hyper_net_unwrapped.transform_structure_vector(
            torch.ones((1, quantizer_unwrapped.vq_embed_dim), device=self.accelerator.device))

        prediction_model_unwrapped.set_structure(arch_vecs_separated)

        model_input = self.vae.encode(batch["pixel_values"][:1].to(self.weight_dtype)).latent_dist.sample()

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2],
            model_input.shape[3],
            self.accelerator.device,
            self.weight_dtype,
        )

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.config.model.prediction_model.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.config.model.prediction_model.logit_mean,
            logit_std=self.config.model.prediction_model.logit_std,
            mode_scale=self.config.model.prediction_model.mode_scale,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

        if self.prediction_model.config.guidance_embeds:
            guidance = torch.tensor([self.config.model.prediction_model.guidance_scale],
                                    device=self.accelerator.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        prompt_embeds = batch["prompt_embeds"][:1].to(self.accelerator.device, dtype=self.weight_dtype)
        pooled_prompt_embdes = batch["pooled_prompt_embeds"][:1].to(self.accelerator.device, dtype=self.weight_dtype)
        text_ids = batch["text_ids"][:1].to(self.accelerator.device)

        macs, params = count_ops_and_params(self.prediction_model,
                                            {'hidden_states': packed_noisy_model_input,
                                             'timestep': timesteps / 1000,
                                             'guidance': guidance,
                                             'pooled_projections': pooled_prompt_embdes,
                                             'encoder_hidden_states': prompt_embeds,
                                             'txt_ids': text_ids,
                                             'img_ids': latent_image_ids,
                                             'return_dict': False,
                                             })

        logger.info(
            "{}'s Params/MACs calculated by OpCounter:\tparams: {:.3f}G\t MACs: {:.3f}G".format(
                self.prediction_model_name, params / 1e9, macs / 1e9))

        sanity_macs_dict = prediction_model_unwrapped.calc_macs()
        prunable_macs_list = [[e / sanity_macs_dict['prunable_macs'] for e in elem] for elem in
                              prediction_model_unwrapped.get_prunable_macs()]

        prediction_model_unwrapped.prunable_macs_list = prunable_macs_list
        prediction_model_unwrapped.resource_info_dict = sanity_macs_dict

        quantizer_unwrapped.set_prunable_macs_template(prunable_macs_list)

        sanity_string = "Our MACs calculation:\t"
        for k, v in sanity_macs_dict.items():
            if isinstance(v, torch.Tensor):
                sanity_string += f" {k}: {v.item() / 1e9:.3f}\t"
            else:
                sanity_string += f" {k}: {v / 1e9:.3f}\t"
        logger.info(sanity_string)

    def get_pipeline(self):
        self.init_weight_dtype()
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder2.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.hyper_net.to(self.accelerator.device, dtype=self.weight_dtype)
        self.quantizer.to(self.accelerator.device, dtype=self.weight_dtype)

        if len(self.accelerator.trackers) == 0:
            self.init_trackers()

        if self.hyper_net:
            self.hyper_net.eval()
        if self.quantizer:
            self.quantizer.eval()
        pipeline = FluxPruningPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            vae=self.accelerator.unwrap_model(self.vae),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            transformer=self.accelerator.unwrap_model(self.prediction_model),
            torch_dtype=self.weight_dtype,
            hyper_net=self.accelerator.unwrap_model(self.hyper_net),
            quantizer=self.accelerator.unwrap_model(self.quantizer),
        )
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=not self.accelerator.is_main_process)

        if self.config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

    @torch.no_grad()
    def generate_samples_from_prompts(self, pretrain=False):
        logger.info("Generating samples from the given prompts... ")

        pipeline = self.get_pipeline()

        images = []
        prompts_resource_ratios = []

        for step, batch in enumerate(self.prompt_dataloader):
            if self.config.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
            gen_images, _, resource_ratios = pipeline(batch["prompts"],
                                                      num_inference_steps=self.config.training.num_inference_steps,
                                                      generator=generator, output_type="pt",
                                                      max_sequence_length=self.config.model.tokenizer.max_sequence_length,
                                                      return_mapped_indices=True,
                                                      hyper_net_input=batch["mpnet_embeddings"].to(self.accelerator.device, dtype=self.weight_dtype),
                                                      pretrain=pretrain
                                                      )
            gen_images = gen_images.images
            gen_images = self.accelerator.gather_for_metrics(gen_images)
            resource_ratios = self.accelerator.gather_for_metrics(resource_ratios)
            images += gen_images
            prompts_resource_ratios += resource_ratios

        images = [torchvision.transforms.ToPILImage()(img.to(dtype=torch.float32)) for img in images]

        imgs_len = len(images)
        n_cols = 4 if imgs_len % 4 == 0 else 3 if imgs_len % 3 == 0 else 2 if imgs_len % 2 == 0 else 1
        prompts_resource_ratios = torch.cat(prompts_resource_ratios, dim=0).cpu().numpy()
        prompts_resource_ratios_images = create_heatmap(prompts_resource_ratios, n_rows=n_cols,
                                                        n_cols=len(prompts_resource_ratios) // n_cols)
        images = make_image_grid(images, n_cols, len(images) // n_cols)

        self.accelerator.log(
            {
                "images/prompt images": wandb.Image(images),
                "images/prompts resource ratio heatmap": wandb.Image(prompts_resource_ratios_images),
            },
            log_kwargs={"wandb": {"commit": False}}
        )
        return images

    @torch.no_grad()
    def log_quantizer_embedding_samples(self, step, save_to_disk=False):
        logger.info("Skipping logging quantizer embedding samples for FluxPruner")


class UnetFineTuner(Trainer):
    def __init__(self, config: DictConfig):
        self.prediction_model_name = "unet"
        super().__init__(config)
        self.hyper_net, self.quantizer, self.mpnet_model, self.mpnet_tokenizer = None, None, None, None

    def init_models(self):
        logger.info("Loading models...")

        # Load scheduler, tokenizer and models.
        noise_scheduler = DDIMScheduler.from_pretrained(self.config.pretrained_model_name_or_path,
                                                        subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.config.revision
        )

        hyper_net = HyperStructure.from_pretrained(self.config.pruning_ckpt_dir, subfolder="hypernet")
        quantizer = StructureVectorQuantizer.from_pretrained(self.config.pruning_ckpt_dir, subfolder="quantizer")

        mpnet_tokenizer = AutoTokenizer.from_pretrained(self.config.prompt_encoder_model_name_or_path)
        mpnet_model = AutoModel.from_pretrained(self.config.prompt_encoder_model_name_or_path)

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.config.revision
            )
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="vae", revision=self.config.revision
            )
        teacher_model = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.prediction_model_name,
            revision=self.config.revision,
        )

        sample_inputs = {'sample': torch.randn(1, teacher_model.config.in_channels, teacher_model.config.sample_size,
                                               teacher_model.config.sample_size),
                         'timestep': torch.ones((1,)).long(),
                         'encoder_hidden_states': text_encoder(torch.tensor([[100]]))[0],
                         }

        teacher_macs, teacher_params = count_ops_and_params(teacher_model, sample_inputs)

        embeddings_gs = torch.load(os.path.join(self.config.pruning_ckpt_dir, "quantizer_embeddings.pt"),
                                   map_location="cpu")
        arch_v = embeddings_gs[self.config.expert_id % embeddings_gs.shape[0]].unsqueeze(0)

        torch.save(arch_v, os.path.join(self.config.training.logging.logging_dir, "arch_vector.pt"))

        unet = UNet2DConditionModelPruned.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.prediction_model_name,
            revision=self.config.non_ema_revision,
            down_block_types=tuple(self.config.model.prediction_model.unet_down_blocks),
            mid_block_type=self.config.model.prediction_model.unet_mid_block,
            up_block_types=tuple(self.config.model.prediction_model.unet_up_blocks),
            gated_ff=self.config.model.prediction_model.gated_ff,
            ff_gate_width=self.config.model.prediction_model.ff_gate_width,
            arch_vector=arch_v,
            random_init=self.config.model.prediction_model.get("random_init", False)
        )

        unet_macs, unet_params = count_ops_and_params(unet, sample_inputs)

        logger.info(f"Teacher macs: {teacher_macs / 1e9}G, Teacher Params: {teacher_params / 1e6}M")
        logger.info(
            f"Pruned {self.prediction_model_name} macs: {unet_macs / 1e9}G, Pruned {self.prediction_model_name} Params: {unet_params / 1e6}M")
        logger.info(f"Pruning Raio: {unet_macs / teacher_macs:.2f}")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        teacher_model.requires_grad_(False)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.mpnet_tokenizer = mpnet_tokenizer
        self.mpnet_model = mpnet_model
        self.teacher_model = teacher_model
        self.prediction_model = unet
        self.hyper_net = hyper_net
        self.quantizer = quantizer

    def init_datasets(self):
        logger.info("Loading datasets...")
        dataset = get_dataset(self.config.data)
        dataset_name = self.config.data.dataset_name

        column_names = dataset["train"].column_names
        caption_column = self.config.data.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column '{self.config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

        if self.config.data.get("filter_dataset", True):
            train_mapped_indices_path = os.path.join(self.config.pruning_ckpt_dir,
                                                     f"{dataset_name}_train_mapped_indices.pt")
            validation_mapped_indices_path = os.path.join(self.config.pruning_ckpt_dir,
                                                          f"{dataset_name}validation_mapped_indices.pt")
            if os.path.exists(train_mapped_indices_path) and os.path.exists(validation_mapped_indices_path):
                logging.info("Skipping filtering dataset. Loading indices from disk.")
                tr_indices = torch.load(train_mapped_indices_path, map_location="cpu")
                val_indices = torch.load(validation_mapped_indices_path, map_location="cpu")
            else:
                tr_indices, val_indices = filter_dataset(dataset, self.hyper_net, self.quantizer, self.mpnet_model,
                                                         self.mpnet_tokenizer, caption_column=caption_column)

            filtered_train_indices = torch.where(tr_indices == self.config.expert_id)[0]
            filtered_validation_indices = torch.where(val_indices == self.config.expert_id)[0]

            dataset["train"] = dataset["train"].select(filtered_train_indices)
            dataset["validation"] = dataset["validation"].select(filtered_validation_indices)

        return dataset

    def init_optimizer(self):
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        scaling_factor = (self.config.training.gradient_accumulation_steps *
                          self.config.data.dataloader.train_batch_size * self.accelerator.num_processes)

        if self.config.training.optim.scale_lr:
            self.config.training.optim.prediction_model_learning_rate = (
                    self.config.training.optim.prediction_model_learning_rate * math.sqrt(scaling_factor)
            )

        params = [
            {"params": [p for p in self.prediction_model.parameters()],
             "lr": self.config.training.optim.prediction_model_learning_rate,
             "weight_decay": self.config.training.optim.prediction_model_weight_decay},
        ]

        optimizer = self.get_optimizer(params)
        return optimizer

    def init_losses(self):
        self.ddpm_loss = F.mse_loss
        self.distillation_loss = F.mse_loss

    def prepare_with_accelerator(self):
        (self.prediction_model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.prompt_dataloader,
         self.lr_scheduler) = (self.accelerator.prepare(self.prediction_model, self.optimizer, self.train_dataloader,
                                                        self.eval_dataloader, self.prompt_dataloader,
                                                        self.lr_scheduler))

    def init_dataloaders(self, data_collate_fn, prompts_collate_fn):
        train_dataloader, eval_dataloader, prompt_dataloader = self.get_main_dataloaders(data_collate_fn,
                                                                                         prompts_collate_fn)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.prompt_dataloader = prompt_dataloader

    def train(self):
        self.init_weight_dtype()

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.config.training.losses.block_loss.weight > 0 or self.config.training.losses.distillation_loss.weight > 0:
            self.teacher_model.to(self.accelerator.device, dtype=self.weight_dtype)

        self.update_train_steps()

        initial_global_step, first_epoch = self.load_checkpoint()
        global_step = initial_global_step

        # Train!
        logging_dir = self.config.training.logging.logging_dir
        total_batch_size = (self.config.data.dataloader.train_batch_size * self.accelerator.num_processes *
                            self.config.training.gradient_accumulation_steps)

        logger.info("***** Running finetuning *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.config.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.data.dataloader.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.training.max_train_steps}")

        progress_bar = tqdm(
            range(0, self.config.training.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )

        self.block_act_teacher = {}
        self.block_act_student = {}
        self.cast_block_act_hooks(self.prediction_model, self.block_act_student)
        self.cast_block_act_hooks(self.teacher_model, self.block_act_teacher)

        for epoch in range(first_epoch, self.config.training.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                if batch["pixel_values"].numel() == 0:
                    continue
                train_loss = 0.0
                self.teacher_model.eval()
                self.prediction_model.train()

                loss, diff_loss, distillation_loss, block_loss = self.step(batch)
                avg_loss = loss
                train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

                # Back-propagate
                self.accelerator.backward(loss)

                if self.config.training.optim.get("clip_grad_norm", None):
                    self.accelerator.clip_grad_norm_(self.prediction_model.parameters(),
                                                     self.config.training.optim.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    log_dict = {
                        "finetuning/loss": train_loss,
                        "finetuning/diffusion_loss": diff_loss,
                        "finetuning/distillation_loss": distillation_loss.detach().item(),
                        "finetuning/block_loss": block_loss.detach().item(),
                        "finetuning/prediction_model_lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    self.accelerator.log(log_dict)

                    logs = {
                        "step_loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % self.config.training.validation_steps == 0:
                        if self.eval_dataset is not None:
                            self.validate()

                    if (global_step % self.config.training.image_logging_steps == 0 or
                            (epoch == self.config.training.num_train_epochs - 1 and step == len(
                                self.train_dataloader) - 1)):

                        # generate some validation images
                        if self.config.data.prompts is not None:
                            val_images = self.generate_samples_from_prompts()

                    # Save checkpoint
                    if hasattr(self.config.training, "checkpoint_steps") and self.config.training.checkpoint_steps and \
                            global_step % self.config.training.checkpoint_steps == 0 and global_step > 0:
                        if self.accelerator.is_main_process:
                            self.save_checkpoint(logging_dir, global_step)
                            if os.path.exists(os.path.join(logging_dir, "arch_vector.pt")):
                                shutil.copy(os.path.join(logging_dir, "arch_vector.pt"),
                                            os.path.join(logging_dir, f"checkpoint-{global_step}"))

                    global_step += 1

                if global_step >= self.config.training.max_train_steps:
                    break

            # checkpoint at the end of each epoch
            if not hasattr(self.config.training, "checkpoint_steps") or not self.config.training.checkpoint_steps:
                if self.accelerator.is_main_process:
                    self.save_checkpoint(logging_dir, global_step)
                    if os.path.exists(os.path.join(logging_dir, "arch_vector.pt")):
                        shutil.copy(os.path.join(logging_dir, "arch_vector.pt"),
                                    os.path.join(logging_dir, f"checkpoint-{global_step}"))

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if self.config.push_to_hub:
                    self.save_model_card(self.repo_id, val_images, repo_folder=self.config.output_dir)
                    upload_folder(
                        repo_id=self.repo_id,
                        folder_path=logging_dir,
                        commit_message="End of training",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

        # checkpoint at the end of training
        if self.accelerator.is_main_process:
            self.save_checkpoint(logging_dir, global_step)
            if os.path.exists(os.path.join(logging_dir, "arch_vector.pt")):
                shutil.copy(os.path.join(logging_dir, "arch_vector.pt"),
                            os.path.join(logging_dir, f"checkpoint-{global_step}"))

        self.accelerator.end_training()

    def step(self, batch):
        # This is similar to the Pruner step function. Functions were not extracted to maintain clarity and readability.
        latents = self.vae.encode(batch["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.config.model.prediction_model.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.config.model.prediction_model.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if self.config.model.prediction_model.input_perturbation:
            new_noise = noise + self.config.model.prediction_model.input_perturbation * torch.randn_like(noise)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        if self.config.model.prediction_model.max_scheduler_steps is None:
            self.config.model.prediction_model.max_scheduler_steps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, self.config.model.prediction_model.max_scheduler_steps, (bsz,),
                                  device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward process)
        if self.config.model.prediction_model.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = batch["prompt_embeds"].to(self.accelerator.device, dtype=self.weight_dtype)

        # Get the target for loss depending on the prediction type
        if self.config.model.prediction_model.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.config.model.prediction_model.prediction_type)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        if self.config.training.losses.block_loss.weight > 0 or self.config.training.losses.distillation_loss.weight > 0:
            with torch.no_grad():
                full_model_pred = self.teacher_model(noisy_latents, timesteps, encoder_hidden_states).sample.detach()
        # Predict the noise residual and compute loss
        model_pred = self.prediction_model(noisy_latents, timesteps, encoder_hidden_states).sample
        if self.config.training.losses.diffusion_loss.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                    torch.stack(
                        [snr,
                         self.config.training.losses.diffusion_loss.snr_gamma * torch.ones_like(timesteps)],
                        dim=1).min(dim=1)[0] / snr
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        diff_loss = loss.clone().detach().mean()
        loss *= self.config.training.losses.diffusion_loss.weight

        block_loss = torch.tensor(0.0, device=self.accelerator.device)
        if self.config.training.losses.block_loss.weight > 0:
            for key in self.block_act_student.keys():
                block_loss += F.mse_loss(self.block_act_student[key], self.block_act_teacher[key].detach(),
                                         reduction="mean")
            block_loss /= len(self.block_act_student)
            loss += self.config.training.losses.block_loss.weight * block_loss

        distillation_loss = torch.tensor(0.0, device=self.accelerator.device)
        if self.config.training.losses.distillation_loss.weight > 0:
            distillation_loss = F.mse_loss(model_pred.float(), full_model_pred.float(), reduction="mean")
            loss += self.config.training.losses.distillation_loss.weight * distillation_loss

        return loss, diff_loss, distillation_loss, block_loss

    @torch.no_grad()
    def validate(self):
        self.init_weight_dtype()
        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if len(self.accelerator.trackers) == 0:
            self.init_trackers()

        progress_bar = tqdm(
            range(0, len(self.eval_dataloader)),
            initial=0,
            desc="Val Steps",
            disable=not self.accelerator.is_main_process,
        )

        self.prediction_model.eval()

        total_val_loss, total_diff_loss, total_distillation_loss, total_block_loss = 0.0, 0.0, 0.0, 0.0

        for step, batch in enumerate(self.eval_dataloader):
            if batch["pixel_values"].numel() == 0:
                continue
            loss, diff_loss, distillation_loss, block_loss = self.step(batch)
            total_val_loss += loss.item()
            total_diff_loss += diff_loss.item()
            total_distillation_loss += distillation_loss.item()
            total_block_loss += block_loss.item()
            progress_bar.update(1)

        total_val_loss /= len(self.eval_dataloader)
        total_diff_loss /= len(self.eval_dataloader)
        total_distillation_loss /= len(self.eval_dataloader)
        total_block_loss /= len(self.eval_dataloader)

        total_val_loss = self.accelerator.reduce(torch.tensor(total_val_loss, device=self.accelerator.device),
                                                 "mean").item()
        total_diff_loss = self.accelerator.reduce(torch.tensor(total_diff_loss, device=self.accelerator.device),
                                                  "mean").item()
        total_distillation_loss = self.accelerator.reduce(torch.tensor(total_distillation_loss,
                                                                       device=self.accelerator.device),
                                                          "mean").item()
        total_block_loss = self.accelerator.reduce(torch.tensor(total_block_loss, device=self.accelerator.device),
                                                   "mean").item()

        self.accelerator.log({
            "validation/loss": total_val_loss,
            "validation/diffusion_loss": total_diff_loss,
            "validation/distillation_loss": total_distillation_loss,
            "validation/block_loss": total_block_loss
        },
            log_kwargs={"wandb": {"commit": False}})

    @torch.no_grad()
    def generate_samples_from_prompts(self):
        logger.info("Generating samples from the given prompts... ")

        pipeline = self.get_pipeline()
        images = []

        for step, batch in enumerate(self.prompt_dataloader):
            if self.config.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
            gen_images = pipeline.generate_samples(batch["prompts"],
                                                   num_inference_steps=self.config.training.num_inference_steps,
                                                   generator=generator, output_type="pt"
                                                   ).images
            gen_images = self.accelerator.gather_for_metrics(gen_images)
            images += gen_images

        images = [torchvision.transforms.ToPILImage()(img) for img in images]
        imgs_len = len(images)
        n_cols = 4 if imgs_len % 4 == 0 else 3 if imgs_len % 3 == 0 else 2 if imgs_len % 2 == 0 else 1
        images = make_image_grid(images, n_cols, len(images) // n_cols)

        self.accelerator.log(
            {
                "images/prompt images": wandb.Image(images),
            },
            log_kwargs={"wandb": {"commit": False}}
        )

        return images


class BilevelUnetFineTuner(UnetFineTuner):

    def __init__(self, config: DictConfig):
        self.prediction_model_name = "unet"
        self.tokenizer, self.text_encoder, self.vae, self.noise_scheduler, self.mpnet_tokenizer, self.mpnet_model, \
            self.prediction_model, self.hyper_net, self.quantizer = None, None, None, None, None, None, None, None, None
        self.train_dataset, self.eval_dataset, self.prompt_dataset = None, None, None
        (self.train_dataloader, self.eval_dataloader, self.prompt_dataloader,
         self.quantizer_embeddings_dataloader) = None, None, None, None
        self.ddpm_loss, self.distillation_loss, self.resource_loss, self.contrastive_loss = None, None, None, None

        self.config = config

        self.accelerator = self.create_accelerator()
        self.set_multi_gpu_logging()

        self.init_models()
        self.enable_xformers()
        self.enable_grad_checkpointing()

        dataset = self.init_datasets()
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["validation"]

        # used for sampling during pruning
        if self.config.data.prompts is None:
            self.config.data.prompts = dataset["validation"][self.config.data.caption_column][
                                       :self.config.data.max_generated_samples]
        self.init_prompts()

        (preprocess_train, preprocess_eval, preprocess_prompts) = self.init_dataset_preprocessors(dataset)
        self.prepare_datasets(preprocess_train, preprocess_eval, preprocess_prompts)
        self.init_dataloaders(collate_fn, prompts_collator)

        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()

        self.init_losses()

        self.configure_logging()
        self.create_logging_dir()

        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            self.init_accelerate_customized_saving_hooks()

        self.overrode_max_train_steps = False
        self.update_config_params()

        self.upper_dataset = self.init_upper_dataset(preprocess_train)
        self.upper_dataloader = self.init_upper_dataloaders(collate_fn)
        self.upper_optimizer = self.init_upper_optimizer()
        self.upper_lr_scheduler = self.init_upper_lr_scheduler()

        self.prepare_with_accelerator()

        self.hyper_net, self.quantizer, self.mpnet_model, self.mpnet_tokenizer = None, None, None, None

    def init_upper_dataset(self, preprocess_train):
        logger.info("Loading upper dataset...")
        dataset = load_dataset(self.config.upper_data.dataset_name)

        column_names = dataset["train"].column_names
        caption_column = self.config.upper_data.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column '{self.config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

        style = self.config.upper_data.get("style", None)

        if style is not None:
            dataset['train'] = dataset['train'].filter(lambda x: x['style'] in style)

        return dataset['train'].with_transform(preprocess_train)

    def init_upper_dataloaders(self, data_collate_fn):
        upper_dataloader = self.get_upper_dataloader(data_collate_fn)
        return upper_dataloader

    def get_upper_dataloader(self, data_collate_fn):
        upper_dataloader = torch.utils.data.DataLoader(
            self.upper_dataset,
            shuffle=True,
            collate_fn=data_collate_fn,
            batch_size=self.config.data.dataloader.train_batch_size,
            num_workers=self.config.data.dataloader.dataloader_num_workers,
        )
        return upper_dataloader

    def init_upper_lr_scheduler(self):
        lr_scheduler = get_scheduler(
            self.config.training.optim.get("upper_lr_scheduler", self.config.training.optim.lr_scheduler),
            optimizer=self.upper_optimizer,
            num_warmup_steps=self.config.training.optim.get("upper_lr_warmup_steps",
                                                            self.config.training.optim.lr_warmup_steps) * self.accelerator.num_processes,
            num_training_steps=self.config.training.max_train_steps * self.accelerator.num_processes // self.config.training.upper_step_freq
        )
        return lr_scheduler

    def init_upper_optimizer(self):
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        scaling_factor = (self.config.training.gradient_accumulation_steps *
                          self.config.data.dataloader.train_batch_size * self.accelerator.num_processes)

        if self.config.training.optim.scale_lr:
            self.config.training.optim.prediction_model_upper_learning_rate = (
                    self.config.training.optim.prediction_model_upper_learning_rate * math.sqrt(scaling_factor)
            )

        params = [
            {"params": [p for p in self.prediction_model.parameters()],
             "lr": self.config.training.optim.prediction_model_upper_learning_rate,
             "weight_decay": self.config.training.optim.prediction_model_weight_decay},
        ]

        optimizer = self.get_upper_optimizer(params)
        return optimizer

    def get_upper_optimizer(self, params):
        if self.config.training.optim.get("upper_optimizer", self.config.training.optim.optimizer) == "adamw":
            # Initialize the optimizer
            if self.config.training.optim.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                    )

                optimizer_cls = bnb.optim.AdamW8bit
            else:
                optimizer_cls = torch.optim.AdamW
            optimizer = optimizer_cls(
                params,
                betas=(self.config.training.optim.get("upper_adam_beta1", self.config.training.optim.adam_beta1),
                       self.config.training.optim.get("upper_adam_beta2", self.config.training.optim.adam_beta2)),
                eps=self.config.training.optim.get("upper_adam_epsilon", self.config.training.optim.adam_epsilon),
            )
            return optimizer
        else:
            raise ValueError(f"Unknown upper optimizer {self.config.training.optim.optimizer}")

    def prepare_with_accelerator(self):
        (self.prediction_model, self.optimizer, self.upper_optimizer, self.train_dataloader, self.eval_dataloader,
         self.prompt_dataloader, self.upper_dataloader, self.lr_scheduler, self.upper_lr_scheduler) = (
            self.accelerator.prepare(self.prediction_model, self.optimizer, self.upper_optimizer,
                                     self.train_dataloader, self.eval_dataloader, self.prompt_dataloader,
                                     self.upper_dataloader, self.lr_scheduler, self.upper_lr_scheduler))

    def train(self):
        self.init_weight_dtype()

        # Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.config.training.losses.block_loss.weight > 0 or self.config.training.losses.distillation_loss.weight > 0:
            self.teacher_model.to(self.accelerator.device, dtype=self.weight_dtype)

        self.update_train_steps()

        initial_global_step, first_epoch = self.load_checkpoint()
        global_step = initial_global_step

        # Train!
        logging_dir = self.config.training.logging.logging_dir
        total_batch_size = (self.config.data.dataloader.train_batch_size * self.accelerator.num_processes *
                            self.config.training.gradient_accumulation_steps)

        logger.info("***** Running finetuning *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Upper examples = {len(self.upper_dataset)}")
        logger.info(f"  Num Epochs = {self.config.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.data.dataloader.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.training.max_train_steps}")

        progress_bar = tqdm(
            range(0, self.config.training.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )

        self.block_act_teacher = {}
        self.block_act_student = {}

        self.cast_block_act_hooks(self.prediction_model, self.block_act_student)
        self.cast_block_act_hooks(self.teacher_model, self.block_act_teacher)

        upper_dataloader_iter = iter(self.upper_dataloader)

        for epoch in range(first_epoch, self.config.training.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                if batch["pixel_values"].numel() == 0:
                    continue
                train_loss = 0.0
                self.teacher_model.eval()
                self.prediction_model.train()

                loss, diff_loss, distillation_loss, block_loss = self.step(batch)
                avg_loss = loss
                train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

                # Back-propagate
                self.accelerator.backward(loss)

                if self.config.training.optim.get("clip_grad_norm", None):
                    self.accelerator.clip_grad_norm_(self.prediction_model.parameters(),
                                                     self.config.training.optim.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:

                    if (global_step + 1) % self.config.training.upper_step_freq == 0:
                        logger.info("Upper step")

                        # sample a batch from the upper dataset
                        try:
                            upper_batch = next(upper_dataloader_iter)
                        except StopIteration:
                            upper_dataloader_iter = iter(self.upper_dataloader)
                            upper_batch = next(upper_dataloader_iter)

                        upper_loss, upper_diff_loss, upper_distillation_loss, upper_block_loss = self.upper_step(
                            upper_batch)

                        self.accelerator.backward(upper_loss)

                        if self.config.training.optim.get("clip_grad_norm", None):
                            self.accelerator.clip_grad_norm_(self.prediction_model.parameters(),
                                                             self.config.training.optim.max_grad_norm)

                        self.upper_optimizer.step()
                        self.upper_lr_scheduler.step()
                        self.upper_optimizer.zero_grad()

                    progress_bar.update(1)
                    log_dict = {
                        "finetuning/loss": train_loss,
                        "finetuning/diffusion_loss": diff_loss,
                        "finetuning/distillation_loss": distillation_loss.detach().item(),
                        "finetuning/block_loss": block_loss.detach().item(),
                        "finetuning/prediction_model_lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    if (global_step + 1) % self.config.training.upper_step_freq == 0:
                        log_dict.update({
                            "finetuning/upper_loss": upper_loss.detach().item(),
                            "finetuning/upper_diffusion_loss": upper_diff_loss.detach().item(),
                            "finetuning/upper_distillation_loss": upper_distillation_loss.detach().item(),
                            "finetuning/upper_block_loss": upper_block_loss.detach().item(),
                            "finetuning/upper_prediction_model_lr": self.upper_lr_scheduler.get_last_lr()[0],
                        })
                    self.accelerator.log(log_dict)

                    logs = {
                        "step_loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    if (global_step + 1) % self.config.training.upper_step_freq == 0:
                        logs.update({
                            "upper_step_loss": upper_loss.detach().item(),
                            "upper_lr": self.upper_lr_scheduler.get_last_lr()[0],
                        })

                    progress_bar.set_postfix(**logs)

                    if global_step % self.config.training.validation_steps == 0:
                        if self.eval_dataset is not None:
                            self.validate()

                    if (global_step % self.config.training.image_logging_steps == 0 or
                            (epoch == self.config.training.num_train_epochs - 1 and step == len(
                                self.train_dataloader) - 1)):

                        torch.cuda.empty_cache()

                        # generate some validation images
                        if self.config.data.prompts is not None:
                            val_images = self.generate_samples_from_prompts()

                    # Save checkpoint
                    if hasattr(self.config.training, "checkpoint_steps") and self.config.training.checkpoint_steps and \
                            global_step % self.config.training.checkpoint_steps == 0 and global_step > 0:
                        if self.accelerator.is_main_process:
                            self.save_checkpoint(logging_dir, global_step)
                            if os.path.exists(os.path.join(logging_dir, "arch_vector.pt")):
                                shutil.copy(os.path.join(logging_dir, "arch_vector.pt"),
                                            os.path.join(logging_dir, f"checkpoint-{global_step}"))

                    global_step += 1

                if global_step >= self.config.training.max_train_steps:
                    break

            # checkpoint at the end of each epoch
            if not hasattr(self.config.training, "checkpoint_steps") or not self.config.training.checkpoint_steps:
                if self.accelerator.is_main_process:
                    self.save_checkpoint(logging_dir, global_step)
                    if os.path.exists(os.path.join(logging_dir, "arch_vector.pt")):
                        shutil.copy(os.path.join(logging_dir, "arch_vector.pt"),
                                    os.path.join(logging_dir, f"checkpoint-{global_step}"))

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if self.config.push_to_hub:
                    self.save_model_card(self.repo_id, val_images, repo_folder=self.config.output_dir)
                    upload_folder(
                        repo_id=self.repo_id,
                        folder_path=logging_dir,
                        commit_message="End of training",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

        # checkpoint at the end of training
        if self.accelerator.is_main_process:
            self.save_checkpoint(logging_dir, global_step)
            if os.path.exists(os.path.join(logging_dir, "arch_vector.pt")):
                shutil.copy(os.path.join(logging_dir, "arch_vector.pt"),
                            os.path.join(logging_dir, f"checkpoint-{global_step}"))

        self.accelerator.end_training()

    def upper_step(self, batch):
        # This is similar to the Pruner step function. Functions were not extracted to maintain clarity and readability.
        latents = self.vae.encode(batch["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.config.model.prediction_model.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.config.model.prediction_model.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if self.config.model.prediction_model.input_perturbation:
            new_noise = noise + self.config.model.prediction_model.input_perturbation * torch.randn_like(noise)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        if self.config.model.prediction_model.max_scheduler_steps is None:
            self.config.model.prediction_model.max_scheduler_steps = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, self.config.model.prediction_model.max_scheduler_steps, (bsz,),
                                  device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward process)
        if self.config.model.prediction_model.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = batch["prompt_embeds"].to(self.accelerator.device, dtype=self.weight_dtype)
        empty_encoder_hidden_states = batch["empty_prompt_embeds"].to(self.accelerator.device, dtype=self.weight_dtype)

        # Get the target for loss depending on the prediction type
        if self.config.model.prediction_model.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.config.model.prediction_model.prediction_type)

        # if self.noise_scheduler.config.prediction_type == "epsilon":
        #     target = noise
        # elif self.noise_scheduler.config.prediction_type == "v_prediction":
        #     target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        # else:
        #     raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.config.training.losses.block_loss.weight > 0 or self.config.training.losses.distillation_loss.weight > 0:
            with torch.no_grad():
                full_model_pred_cond = self.teacher_model(noisy_latents, timesteps,
                                                          encoder_hidden_states).sample.detach()
                full_model_pred_uncond = self.teacher_model(noisy_latents, timesteps,
                                                            empty_encoder_hidden_states).sample.detach()

        # Predict the noise residual and compute loss
        model_pred = self.prediction_model(noisy_latents, timesteps, encoder_hidden_states).sample

        # if self.config.training.losses.diffusion_loss.snr_gamma is None:
        #     loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        # else:
        #     # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        #     # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        #     # This is discussed in Section 4.2 of the same paper.
        #     snr = compute_snr(self.noise_scheduler, timesteps)
        #     if self.noise_scheduler.config.prediction_type == "v_prediction":
        #         # Velocity objective requires that we add one to SNR values before we divide by them.
        #         snr = snr + 1
        #     mse_loss_weights = (
        #             torch.stack(
        #                 [snr,
        #                  self.config.training.losses.diffusion_loss.snr_gamma * torch.ones_like(timesteps)],
        #                 dim=1).min(dim=1)[0] / snr
        #     )
        #
        #     loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        #     loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        #     loss = loss.mean()

        # diff_loss = loss.clone().detach().mean()
        # loss *= self.config.training.losses.diffusion_loss.upper_weight

        diff_loss = torch.tensor(0.0, device=self.accelerator.device)
        loss = 0.0

        block_loss = torch.tensor(0.0, device=self.accelerator.device)
        if self.config.training.losses.block_loss.upper_weight > 0:
            for key in self.block_act_student.keys():
                block_loss += F.mse_loss(self.block_act_student[key], self.block_act_teacher[key].detach(),
                                         reduction="mean")
            block_loss /= len(self.block_act_student)
            loss += self.config.training.losses.block_loss.upper_weight * block_loss

        distillation_loss = torch.tensor(0.0, device=self.accelerator.device)
        if self.config.training.losses.distillation_loss.upper_weight > 0:
            distillation_loss = F.mse_loss(model_pred,
                                           (full_model_pred_uncond - (full_model_pred_cond - full_model_pred_uncond)),
                                           reduction="mean")
            loss += self.config.training.losses.distillation_loss.upper_weight * distillation_loss

        return loss, diff_loss, distillation_loss, block_loss


class NudityBilevelUnetFineTuner(BilevelUnetFineTuner):
    def init_upper_dataset(self, preprocess_train):
        logger.info("Loading upper dataset...")
        dataset = load_dataset(self.config.upper_data.dataset_name)

        column_names = dataset["train"].column_names
        caption_column = self.config.upper_data.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column '{self.config.data.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

        return dataset['train'].with_transform(preprocess_train)


class DreamBoothBilevelUnetFineTuner(BilevelUnetFineTuner):

    @staticmethod
    def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
        text_input_ids = input_ids.to(text_encoder.device)

        if text_encoder_use_attention_mask:
            attention_mask = attention_mask.to(text_encoder.device)
        else:
            attention_mask = None

        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds

    def init_upper_dataset(self, preprocess_train):

        def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
            if tokenizer_max_length is not None:
                max_length = tokenizer_max_length
            else:
                max_length = tokenizer.model_max_length

            text_inputs = tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            return text_inputs

        class PromptDataset(torch.utils.data.Dataset):
            """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

            def __init__(self, prompt, num_samples):
                self.prompt = prompt
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, index):
                example = {}
                example["prompt"] = self.prompt
                example["index"] = index
                return example

        class DreamBoothDataset(torch.utils.data.Dataset):
            """
            A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
            It pre-processes the images and the tokenizes prompts.
            """

            def __init__(
                    self,
                    instance_data_root,
                    instance_prompt,
                    tokenizer,
                    class_data_root=None,
                    class_prompt=None,
                    class_num=None,
                    size=512,
                    center_crop=False,
                    encoder_hidden_states=None,
                    class_prompt_encoder_hidden_states=None,
                    tokenizer_max_length=None,
            ):
                self.size = size
                self.center_crop = center_crop
                self.tokenizer = tokenizer
                self.encoder_hidden_states = encoder_hidden_states
                self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
                self.tokenizer_max_length = tokenizer_max_length

                self.instance_data_root = Path(instance_data_root)
                if not self.instance_data_root.exists():
                    raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

                self.instance_images_path = list(Path(instance_data_root).iterdir())
                self.num_instance_images = len(self.instance_images_path)
                self.instance_prompt = instance_prompt
                self._length = self.num_instance_images

                if class_data_root is not None:
                    self.class_data_root = Path(class_data_root)
                    self.class_data_root.mkdir(parents=True, exist_ok=True)
                    self.class_images_path = list(self.class_data_root.iterdir())
                    if class_num is not None:
                        self.num_class_images = min(len(self.class_images_path), class_num)
                    else:
                        self.num_class_images = len(self.class_images_path)
                    self._length = max(self.num_class_images, self.num_instance_images)
                    self.class_prompt = class_prompt
                else:
                    self.class_data_root = None

                self.image_transforms = transforms.Compose(
                    [
                        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )

            def __len__(self):
                return self._length

            def __getitem__(self, index):
                example = {}
                instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
                instance_image = exif_transpose(instance_image)

                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
                example["instance_images"] = self.image_transforms(instance_image)

                if self.encoder_hidden_states is not None:
                    example["instance_prompt_ids"] = self.encoder_hidden_states
                else:
                    text_inputs = tokenize_prompt(
                        self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
                    )
                    example["instance_prompt_ids"] = text_inputs.input_ids
                    example["instance_attention_mask"] = text_inputs.attention_mask

                if self.class_data_root:
                    class_image = Image.open(self.class_images_path[index % self.num_class_images])
                    class_image = exif_transpose(class_image)

                    if not class_image.mode == "RGB":
                        class_image = class_image.convert("RGB")
                    example["class_images"] = self.image_transforms(class_image)

                    if self.class_prompt_encoder_hidden_states is not None:
                        example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
                    else:
                        class_text_inputs = tokenize_prompt(
                            self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                        )
                        example["class_prompt_ids"] = class_text_inputs.input_ids
                        example["class_attention_mask"] = class_text_inputs.attention_mask

                return example

        if self.config.training.dreambooth.with_prior_preservation:
            class_images_dir = Path(self.config.training.dreambooth.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.config.training.dreambooth.num_class_images:
                torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
                if self.config.training.dreambooth.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif self.config.training.dreambooth.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif self.config.training.dreambooth.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=self.config.revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.config.training.dreambooth.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.config.training.dreambooth.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                                                                batch_size=self.config.training.dreambooth.sample_batch_size)

                sample_dataloader = self.accelerator.prepare(sample_dataloader)
                pipeline.to(self.accelerator.device)

                for example in tqdm(
                        sample_dataloader, desc="Generating class images",
                        disable=not self.accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if self.config.training.dreambooth.pre_compute_text_embeddings:

            def compute_text_embeddings(prompt):
                with torch.no_grad():
                    text_inputs = tokenize_prompt(self.tokenizer, prompt,
                                                  tokenizer_max_length=self.tokenizer.model_max_length)
                    prompt_embeds = self.encode_prompt(
                        self.text_encoder,
                        text_inputs.input_ids,
                        text_inputs.attention_mask,
                    )

                return prompt_embeds

            pre_computed_encoder_hidden_states = compute_text_embeddings(
                self.config.training.dreambooth.instance_prompt)

            if self.config.training.dreambooth.class_prompt is not None:
                pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(
                    self.config.training.dreambooth.class_prompt)
            else:
                pre_computed_class_prompt_encoder_hidden_states = None

            gc.collect()
            torch.cuda.empty_cache()
        else:
            pre_computed_encoder_hidden_states = None
            pre_computed_class_prompt_encoder_hidden_states = None

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=self.config.training.dreambooth.instance_data_dir,
            instance_prompt=self.config.training.dreambooth.instance_prompt,
            class_data_root=self.config.training.dreambooth.class_data_dir if self.config.training.dreambooth.with_prior_preservation else None,
            class_prompt=self.config.training.dreambooth.class_prompt,
            class_num=self.config.training.dreambooth.num_class_images,
            tokenizer=self.tokenizer,
            size=self.config.training.dreambooth.resolution,
            center_crop=self.config.training.dreambooth.center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
            tokenizer_max_length=self.tokenizer.model_max_length,
        )

        return train_dataset

    def init_upper_dataloaders(self, data_collate_fn):

        return self.get_upper_dataloader()

    def get_upper_dataloader(self, collate_fn=None):

        def upper_collate_fn(examples, with_prior_preservation=False):
            has_attention_mask = "instance_attention_mask" in examples[0]

            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            if has_attention_mask:
                attention_mask = [example["instance_attention_mask"] for example in examples]

            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]

                if has_attention_mask:
                    attention_mask += [example["class_attention_mask"] for example in examples]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = torch.cat(input_ids, dim=0)

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }

            if has_attention_mask:
                attention_mask = torch.cat(attention_mask, dim=0)
                batch["attention_mask"] = attention_mask

            return batch

        train_dataloader = torch.utils.data.DataLoader(
            self.upper_dataset,
            shuffle=True,
            collate_fn=lambda examples: upper_collate_fn(examples,
                                                         self.config.training.dreambooth.with_prior_preservation),
            batch_size=self.config.upper_data.dataloader.train_batch_size,
            num_workers=self.config.data.dataloader.dataloader_num_workers,
        )
        return train_dataloader

    def upper_step(self, batch):
        # empty cuda cache
        torch.cuda.empty_cache()

        # This is similar to the Pruner step function. Functions were not extracted to maintain clarity and readability.
        latents = self.vae.encode(batch["pixel_values"].to(self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        if self.config.model.prediction_model.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise = torch.randn_like(latents) + 0.1 * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=latents.device
            )
        else:
            noise = torch.randn_like(latents)

        bsz, channels, height, width = latents.shape

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        if self.config.training.dreambooth.pre_compute_text_embeddings:
            encoder_hidden_states = batch["input_ids"]
        else:
            encoder_hidden_states = self.encode_prompt(
                self.text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
            )

        if self.unwrap_model(self.prediction_model).config.in_channels == channels * 2:
            noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=1)

        if self.config.training.dreambooth.class_labels_conditioning == "timesteps":
            class_labels = timesteps
        else:
            class_labels = None

            # Predict the noise residual
        model_pred = self.prediction_model(
            noisy_latents, timesteps, encoder_hidden_states, class_labels=class_labels, return_dict=False
        )[0]

        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.config.training.dreambooth.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

        # Compute instance loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.config.training.dreambooth.with_prior_preservation:
            # Add the prior loss to the instance loss.
            loss = loss + self.config.training.dreambooth.prior_loss_weight * prior_loss

        diff_loss = torch.tensor(0.0, device=self.accelerator.device)

        block_loss = torch.tensor(0.0, device=self.accelerator.device)
        distillation_loss = torch.tensor(0.0, device=self.accelerator.device)

        torch.cuda.empty_cache()

        return loss, diff_loss, distillation_loss, block_loss


class SingleArchUnetFinetuner(UnetFineTuner):

    def init_models(self):
        logger.info("Loading models...")

        # Load scheduler, tokenizer and models.
        noise_scheduler = DDIMScheduler.from_pretrained(self.config.pretrained_model_name_or_path,
                                                        subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.config.revision
        )

        hyper_net = HyperStructure.from_pretrained(self.config.pruning_ckpt_dir, subfolder="hypernet")
        quantizer = StructureVectorQuantizer.from_pretrained(self.config.pruning_ckpt_dir, subfolder="quantizer")

        mpnet_tokenizer = AutoTokenizer.from_pretrained(self.config.prompt_encoder_model_name_or_path)
        mpnet_model = AutoModel.from_pretrained(self.config.prompt_encoder_model_name_or_path)

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.config.revision
            )
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="vae", revision=self.config.revision
            )
        teacher_model = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.prediction_model,
            revision=self.config.revision,
        )

        sample_inputs = {'sample': torch.randn(1, teacher_model.config.in_channels, teacher_model.config.sample_size,
                                               teacher_model.config.sample_size),
                         'timestep': torch.ones((1,)).long(),
                         'encoder_hidden_states': text_encoder(torch.tensor([[100]]))[0],
                         }
        teacher_macs, teacher_params = count_ops_and_params(teacher_model, sample_inputs)

        arch_v = hyper_net.arch
        arch_v = quantizer.gumbel_sigmoid_trick(arch_v).to("cpu")

        unet = UNet2DConditionModelPruned.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.prediction_model_name,
            revision=self.config.non_ema_revision,
            down_block_types=tuple(self.config.model.prediction_model.unet_down_blocks),
            mid_block_type=self.config.model.prediction_model.unet_mid_block,
            up_block_types=tuple(self.config.model.prediction_model.unet_up_blocks),
            gated_ff=self.config.model.prediction_model.gated_ff,
            ff_gate_width=self.config.model.prediction_model.ff_gate_width,
            arch_vector=arch_v
        )

        unet_macs, unet_params = count_ops_and_params(unet, sample_inputs)

        logger.info(f"Teacher macs: {teacher_macs / 1e9}G, Teacher Params: {teacher_params / 1e6}M")
        logger.info(f"Single Arch Pruned {self.prediction_model_name} macs: {unet_macs / 1e9}G,"
                    f"Single Arch Pruned {self.prediction_model_name} Params: {unet_params / 1e6}M")
        logger.info(f"Pruning Raio: {unet_macs / teacher_macs:.2f}")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        teacher_model.requires_grad_(False)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.mpnet_tokenizer = mpnet_tokenizer
        self.mpnet_model = mpnet_model
        self.teacher_model = teacher_model
        self.prediction_model = unet
        self.hyper_net = hyper_net
        self.quantizer = quantizer

    def init_datasets(self):
        logger.info("Loading datasets...")
        dataset = get_dataset(self.config.data)
        return dataset


class BaselineUnetFineTuner(UnetFineTuner):

    def __init__(self, config: DictConfig, pruning_type="magnitude"):
        assert pruning_type in ["no-pruning", "magnitude", "random", "structural"]
        self.pruning_type = pruning_type
        super().__init__(config)

    def init_models(self):
        logger.info("Loading models...")

        # Load scheduler, tokenizer and models.
        noise_scheduler = DDIMScheduler.from_pretrained(self.config.pretrained_model_name_or_path,
                                                        subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.config.revision
        )

        mpnet_tokenizer = AutoTokenizer.from_pretrained(self.config.prompt_encoder_model_name_or_path)
        mpnet_model = AutoModel.from_pretrained(self.config.prompt_encoder_model_name_or_path)

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.config.revision
            )
            vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_name_or_path, subfolder="vae", revision=self.config.revision
            )

        teacher_model = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=self.prediction_model_name,
            revision=self.config.revision,
        )
        torch.compile(teacher_model, mode="reduce-overhead", fullgraph=True)

        sample_inputs = {'sample': torch.randn(1, teacher_model.config.in_channels, teacher_model.config.sample_size,
                                               teacher_model.config.sample_size),
                         'timestep': torch.ones((1,)).long(),
                         'encoder_hidden_states': text_encoder(torch.tensor([[100]]))[0],
                         }

        teacher_macs, teacher_params = count_ops_and_params(teacher_model, sample_inputs)

        if self.pruning_type == "magnitude":
            unet = UNet2DConditionModelMagnitudePruned.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder=self.prediction_model_name,
                revision=self.config.non_ema_revision,
                target_pruning_rate=self.config.training.pruning_target,
                pruning_method=self.config.training.pruning_method,
                sample_inputs=sample_inputs
            )

            torch.compile(unet, mode="reduce-overhead", fullgraph=True)
        elif self.pruning_type == "structural":
            unet = torch.load(
                os.path.join(self.config.pruning_ckpt_dir, "unet_pruned.pth"),
                map_location="cpu"
            )
        elif self.pruning_type == "no-pruning":
            unet = copy.deepcopy(teacher_model)
            unet.requires_grad_(True)
        else:
            unet = UNet2DConditionModelPruned.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder=self.prediction_model_name,
                revision=self.config.non_ema_revision,
                down_block_types=tuple(self.config.model.prediction_model.unet_down_blocks),
                mid_block_type=self.config.model.prediction_model.unet_mid_block,
                up_block_types=tuple(self.config.model.prediction_model.unet_up_blocks),
                gated_ff=self.config.model.prediction_model.gated_ff,
                ff_gate_width=self.config.model.prediction_model.ff_gate_width,
                random_pruning_ratio=self.config.training.random_pruning_ratio
            )

        unet_macs, unet_params = count_ops_and_params(unet, sample_inputs)

        logging.info(f"Teacher macs: {teacher_macs / 1e9}G, Teacher Params: {teacher_params / 1e6}M")
        logger.info(
            f"Baseline Pruned {self.prediction_model_name} macs: {unet_macs / 1e9}G,"
            f" Baseline Pruned {self.prediction_model_name} Params: {unet_params / 1e6}M")
        logger.info(f"Pruning Raio: {unet_macs / teacher_macs:.2f}")

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        teacher_model.requires_grad_(False)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.teacher_model = teacher_model
        self.prediction_model = unet
        self.mpnet_tokenizer = mpnet_tokenizer
        self.mpnet_model = mpnet_model

    def init_datasets(self):
        logger.info("Loading datasets...")
        dataset = get_dataset(self.config.data)
        return dataset
