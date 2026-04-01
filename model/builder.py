import logging
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from collections import OrderedDict

from model.VIRST import VirstForCausalLM

def build_virst(config, seg_token_idx, model_args, checkpoint=None):
    # train from the videochat model 
    videochat_ckpt = 'checkpoints'

    kwargs = {
        "seg_token_idx":seg_token_idx
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        'checkpoints/videochat',
        config=config,
        device_map = None,
        trust_remote_code=True,
        use_safetensors=True,
        ignore_mismatched_sizes=True,
        model_args = model_args,
        **kwargs
    )

    model.model.initialize_module()
    model._initialize_keyframe_config()

    return model 

def load_checkpoint_virst(model, checkpoint):
    logging.info("Found checkpoint, load weights...")
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Checkpoint %s keys has been loaded. Missing keys: %s, Unexpected keys: %s",
        len(state_dict),
        len(missing),
        len(unexpected),
    )
    if missing:
        logging.info("Missing keys: %s", missing)
    if unexpected:
        logging.info("Unexpected keys: %s", unexpected)
    logging.info("Loading checkpoint done.")
    return model 
