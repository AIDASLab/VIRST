from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import logging
from tqdm import tqdm 
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM

from utils.utils import IGNORE_INDEX
from model.sam2.sam2_virst import build_sam2_virst
from model.videochat.modeling_videochat_flash import VideoChatFlashQwenModel, VideoChatFlashQwenForCausalLM
from model.videochat.modeling_videochat_flash import VideoChatFlashQwenConfig
from model.seg_prompter import SegPrompter
from model.seg_prompter import InitialSegFusion


def dice_loss(
    inputs   : torch.Tensor,
    targets  : torch.Tensor,
    num_masks: float,
    scale    : float =1000,
    eps      : float =1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def select_keyframes(frame_score, max_prop=5, is_evaluation=False):
    T = frame_score.size(0)

    if T == 1:
        cond_idx = torch.tensor([0], device=frame_score.device)
        prop_idx = torch.tensor([], device=frame_score.device)
        return cond_idx, prop_idx

    if is_evaluation:
        num_cond = max(1, T // 4)
        cond_idx = torch.linspace(0, T - 1, steps=num_cond, device=frame_score.device).round().long()
        prop_idx = torch.arange(T, device=frame_score.device)
        return cond_idx, prop_idx

    num_cond = min(3, T)
    cond_idx = torch.randperm(T, device=frame_score.device)[:num_cond]
    cond_idx, _ = cond_idx.sort()

    prop_set = set()
    for k in cond_idx.tolist():
        for i in range(1, max_prop + 1):
            nxt = k + i
            if nxt < T:
                prop_set.add(nxt)
        if k - 1 >= 0:
            prop_set.add(k - 1)
        if k - 2 >= 0:
            prop_set.add(k - 2)

    prop_idx = torch.tensor(sorted(prop_set - set(cond_idx.tolist())), device=frame_score.device)
    return cond_idx, prop_idx


def zero_touch(*modules):
    z = 0.0
    for m in modules:
        for p in m.parameters():
            z = z + p.view(-1)[0] * 0.0
    return z

class VirstConfig(VideoChatFlashQwenConfig):
    model_type = "virst"

class VirstMetaModel():
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.token_dim = 256
        self.num_maskmem = kwargs.pop("max_uncond_frames_in_mem")
        
    def initialize_module(self):
        sam2_cfg = self.config.sam2_config['cfg']
        
        sam2_checkpoint = self.config.sam2_config['checkpoint']

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_module(self.config.sam2_config['cfg_module'], version_base='1.2')

        self.seg_model = build_sam2_virst(sam2_cfg, sam2_checkpoint, num_maskmem=self.num_maskmem) 

        in_dim = self.config.hidden_size
        token_dim = self.token_dim

        self.seg_prompter = SegPrompter(
            in_dim=in_dim,
            token_dim = token_dim
        )
        self.occ_predictor = nn.Linear(token_dim, 1)

        self.initial_seg_fusion_module = InitialSegFusion(
            hidden_dim=in_dim,
            vision_dim=256,
            token_dim=token_dim,
            grid_hw=8,
            num_heads=8
        )

class VirstModel(VirstMetaModel, VideoChatFlashQwenModel):
    config_class =VirstConfig 

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config ,**kwargs)
        self.config.use_cache = False


class VirstForCausalLM(VideoChatFlashQwenForCausalLM):
    config_class = VirstConfig

    def __init__(
        self, 
        config, 
        model_args,
        **kwargs
    ):
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config, **kwargs)

        kwargs['max_uncond_frames_in_mem'] = model_args.max_uncond_frames_in_mem
        self.model = VirstModel(config, **kwargs)
        self.model_args = model_args
        self.ce_loss_weight = model_args.ce_loss_weight
        self.iou_loss_weight = model_args.iou_loss_weight
        self.occ_loss_weight = model_args.occ_loss_weight
        self.bce_loss_weight = model_args.bce_loss_weight
        self.dice_loss_weight = model_args.dice_loss_weight
        self.num_seg_keyframes = model_args.num_seg_keyframes

        self.initial_seg_fusion = model_args.initial_seg_fusion
        self.vis_threshold = 0.5

    def _initialize_keyframe_config(self):
        self.use_seg_prompter = getattr(self.model_args, "use_seg_prompter", True)

        self.max_cond_frames_in_attn = self.model_args.max_cond_frames_in_attn
        self.max_uncond_frames_in_mem = self.model_args.max_uncond_frames_in_mem

        self.model.seg_model.max_cond_frames_in_attn = self.max_cond_frames_in_attn
        self.model.seg_model.num_maskmem = self.max_uncond_frames_in_mem

        logging.info("eval_keyframe_scheme: uniform")
        logging.info("train_keyframe_scheme: random")
        logging.info("max_cond_frames_in_attn: %s", self.max_cond_frames_in_attn)
        logging.info("max_uncond_frames_in_mem: %s", self.max_uncond_frames_in_mem)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images_clip: Optional[torch.FloatTensor] = None,
        images_sam: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        image_ids: Optional[List[int]] = None, # Mapping from each conversation to its corresponding image
        modalities: Optional[List[str]] = ["image"],
        gt_masks: Optional[List[torch.FloatTensor]] = None, # (conv, T_seg, O,H, W)
        generation: Optional[bool] = True, 
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        seg_evaluate = False, 
        video_path: Optional[List[str]] = None,
        frame_ids: Optional[List[int]] = None, # only for Inference 
        use_cond_frames: Optional[bool] = False,
        filename_save: Optional[str] = None, # for visualization
    ) -> Union[Dict, CausalLMOutputWithPast]:

        if image_sizes is None and images_clip is not None:
            image_sizes = [img[0].shape[-2:] for img in images_clip]
        
        image_features = None 

        if inputs_embeds is None:
            (
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                inputs_embeds, 
                labels, 
                image_features
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                labels, 
                images_clip,
                image_ids, 
                modalities, 
                image_sizes
            )
        
        if image_features is not None and inputs_embeds is not None :
            assert image_features.shape[0] == inputs_embeds.shape[0], (
                f"the number of image features and input_embeds supposed to be same, ",
                f"but got {image_features.shape} and {inputs_embeds.shape}"
            )

        labels_seg_masked = None 
        if labels is not None:
            seg_mask = labels == self.seg_token_idx
            labels_seg_masked = labels.masked_fill(seg_mask, IGNORE_INDEX)

        if self.initial_seg_fusion and images_sam is not None:
            B, T_seg, C, H, W = images_sam.shape
            seg_embeds = inputs_embeds[seg_mask].view(B, -1, inputs_embeds.size(-1))

            with torch.no_grad():
                images_sam_input = images_sam.flatten(0,1)
                seg_backbone_out = self.model.seg_model.forward_image(images_sam_input)
                images_sam_feat = seg_backbone_out['vision_features']
                _, _C, _H, _W = images_sam_feat.shape
                images_sam_feat = images_sam_feat.view(B, T_seg, _C, _H, _W)

            seg_fusion = self.model.initial_seg_fusion_module(
                seg_embeds=seg_embeds,
                vision_feats=images_sam_feat
            )

            new_inputs_embeds = inputs_embeds.clone()
            new_inputs_embeds[seg_mask] = seg_fusion.reshape(-1, seg_fusion.size(-1))
            inputs_embeds = new_inputs_embeds
            
        output = super().forward(
            input_ids= input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels_seg_masked,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            modalities= modalities,
            return_dict=return_dict,
        )
        
        if generation:
            return output

        output_hidden_states = output.hidden_states[-1]
        
        
        num_conv, L = labels.size()
        _, _, D = output_hidden_states.shape

        seg_mask = (labels == self.seg_token_idx)
        seg_token_num = seg_mask.sum(dim=1)
        seg_token_max_num = (seg_token_num.max().item())

        pred_embeddings = torch.zeros(num_conv, seg_token_max_num, D, 
                                      device=output_hidden_states.device, 
                                      dtype=output_hidden_states.dtype)

        for conv_idx in range(num_conv):
            output_idx = seg_mask[conv_idx].nonzero(as_tuple=True)[0]
            if output_idx.numel() > 0:
                pred_embeddings[conv_idx, :seg_token_num[conv_idx]] = output_hidden_states[conv_idx, output_idx]

        if images_sam is not None:
            B, T_seg, C, H, W = images_sam.shape

            images_sam_input = images_sam.flatten(0,1)
            seg_backbone_out = self.model.seg_model.forward_image(images_sam_input)
            images_sam_feat = seg_backbone_out['vision_features']
            _, _C, _H, _W = images_sam_feat.shape
            images_sam_feat = images_sam_feat.view(B, T_seg, _C, _H, _W)

        mask_bce  = torch.tensor(0.0, device=pred_embeddings.device, dtype=pred_embeddings.dtype)
        mask_dice = torch.tensor(0.0, device=pred_embeddings.device, dtype=pred_embeddings.dtype)
        iou_loss  = torch.tensor(0.0, device=pred_embeddings.device, dtype=pred_embeddings.dtype)
        occ_loss  = torch.tensor(0.0, device=pred_embeddings.device, dtype=pred_embeddings.dtype)
        count     = torch.tensor(0.0, device=pred_embeddings.device, dtype=pred_embeddings.dtype)

        pred_mask_list = []
        keyframe_indices_list = []

        for conv_id in range(num_conv):
            if seg_token_num[conv_id] == 0:
                continue
            
            else: 
                image_sam_id = image_ids[conv_id]
                pred_embedding_conv = pred_embeddings[conv_id, :seg_token_num[conv_id], :]

                if self.use_seg_prompter:
                    seg_embeddings, frame_attn_scores = self.model.seg_prompter(
                        seg_token = pred_embedding_conv.unsqueeze(0),
                        image_token = images_sam_feat,
                        return_attn = True,
                    )
                else:
                    num_conv, N, _ = 1, pred_embedding_conv.size(0), pred_embedding_conv.size(1)
                    T_seg = images_sam_feat.size(1)
                    token_dim = self.model.seg_prompter.token_dim

                    if pred_embedding_conv.size(-1) != token_dim:
                        seg_token_proj = self.model.seg_prompter.seg_proj(pred_embedding_conv.unsqueeze(0))
                    else:
                        seg_token_proj = pred_embedding_conv.unsqueeze(0)

                    seg_embeddings = seg_token_proj.unsqueeze(2).expand(num_conv, N, T_seg, token_dim).contiguous()
                    frame_attn_scores = torch.zeros(num_conv, T_seg, device=pred_embedding_conv.device)

                gt_masks_conv = gt_masks[conv_id]
                
            
            B = 1
            frame_score = frame_attn_scores.squeeze(0)

            cond_idx, prop_idx = select_keyframes(
                frame_score,
                is_evaluation=seg_evaluate,
            )
            
            if use_cond_frames:
                seg_idx = torch.cat([cond_idx, prop_idx]).unique().sort()[0].long()
            else:
                seg_idx = cond_idx

            keyframe_indices_list.append(seg_idx)
            
            logging.info(f"cond_idx: {cond_idx}, seg_embedding shape: {seg_embeddings.shape}")
            seg_prompt = seg_embeddings[:, :, cond_idx, :]
            
            if seg_evaluate:
                return self.video_propagate(
                    video_path = video_path,
                    frame_ids = frame_ids[conv_id][0],
                    cond_idx = cond_idx,
                    seg_prompts = seg_prompt,
                )

            images_sam_prop = images_sam[image_sam_id][seg_idx]

            local_cond_idx = torch.arange(len(seg_idx), device=seg_idx.device)[
                torch.isin(seg_idx, cond_idx)
            ]

            if use_cond_frames:
                pred_dicts = self.model.seg_model(
                    input = images_sam_prop.unsqueeze(0),
                    seg_prompt = seg_prompt,
                    backbone_out = None,
                    cond_frames = local_cond_idx,
                )

            else:
                pred_dicts = self.model.seg_model(
                    input = images_sam_prop.unsqueeze(0),
                    seg_prompt = seg_prompt,
                    backbone_out = None,
                )

            pred_mask_conv = [] 

            for t, mask_idx in enumerate(seg_idx):
                out = pred_dicts[t]
                
                pred_logits = out.get("pred_masks_high_res", out["pred_masks"])
                pred_logits = pred_logits.squeeze(1).squeeze(0)

                gt_mask = gt_masks_conv[mask_idx,:,:,:].float()

                cur_mask_bce = sigmoid_ce_loss(pred_logits.unsqueeze(0), gt_mask, num_masks=1)
                cur_mask_dice = dice_loss(pred_logits.unsqueeze(0), gt_mask, num_masks=1)

                mask_bce = mask_bce + cur_mask_bce 
                mask_dice = mask_dice + cur_mask_dice

                pred_ious = out.get("multistep_pred_ious")

                pred_mask_bin = (pred_logits.sigmoid() > 0.5).float()
                gt_mask_bin = (gt_mask.squeeze(0) > 0.5).float()

                intersection = (pred_mask_bin * gt_mask_bin).sum()
                union = pred_mask_bin.sum() + gt_mask_bin.sum() - intersection
                real_iou = (intersection / (union + 1e-6)).item()

                pred_iou = pred_ious[0].max().squeeze()
                iou_loss += torch.abs(real_iou - pred_iou)

                pred_occ = out.get("multistep_object_score_logits")
                pred_occ = pred_occ[0].max().squeeze(0)
                pred_prob = pred_occ.sigmoid()
            
                gt_occ = torch.tensor(
                    float(gt_mask.sum() > 0),
                    device=pred_occ.device
                )
                occ_loss += F.binary_cross_entropy_with_logits(pred_occ, gt_occ)
            
                count += 1

                pred_mask_conv.append(pred_logits.sigmoid())

            pred_mask_list.append(torch.stack(pred_mask_conv))
            
        ce_loss = self.ce_loss_weight * output.loss
        mask_bce = self.bce_loss_weight * (mask_bce / (count + 1e-8))
        mask_dice = self.dice_loss_weight * (mask_dice / (count + 1e-8))
        iou_loss = self.iou_loss_weight * (iou_loss / (count + 1e-8))
        occ_loss = self.occ_loss_weight * (occ_loss / (count + 1e-8))
        mask_loss = mask_bce + mask_dice + iou_loss

        loss = ce_loss + mask_loss + iou_loss + occ_loss 
        loss = loss + zero_touch(
            self.model.seg_prompter,
            self.model.occ_predictor,
            self.model.seg_model,
        )

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "iou_loss": iou_loss,
            "occ_loss": occ_loss,
            "mask_bce_loss": mask_bce,
            "mask_dice_loss": mask_dice,
            "mask_loss": mask_loss,
            "pred_mask": pred_mask_list,
            "keyframe_indices": keyframe_indices_list,
            "occ_scores": [],
            "input_ids": input_ids,
            "labels": labels,
        }

    @torch.no_grad()
    def video_propagate(
        self, 
        video_path, 
        frame_ids, 
        cond_idx, 
        seg_prompts,
    ):
        inference_state = self.model.seg_model.init_state(
            video_path = video_path[0],
            offload_video_to_cpu = False,
            offload_state_to_cpu = False,
            async_loading_frames = False, 
        )
        self.model.seg_model.reset_state(inference_state)
        
        cond_video_idx_list = [frame_ids[i] for i in cond_idx.tolist()]
        
        logging.info(f"seg_prompts.shape: {seg_prompts.shape}")
        
        for cond_idx, cond_frame_video_idx in enumerate(cond_video_idx_list):
            cur_seg_prompt = seg_prompts[:, :, cond_idx, :]
            
            logging.info(f"cur_seg_prompt.shape: {cur_seg_prompt.shape}")
            _, out_obj_ids, out_mask_logits = self.model.seg_model.add_new_prompt(
                inference_state=inference_state,
                frame_idx = cond_frame_video_idx,
                obj_id = 0,
                prompts = cur_seg_prompt
            )
            
        video_segments = {} # per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.seg_model.propagate_in_video(
            inference_state,
            start_frame_idx = 0 
            ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments


AutoConfig.register("virst", VirstConfig)
AutoModelForCausalLM.register(VirstConfig, VirstForCausalLM)
