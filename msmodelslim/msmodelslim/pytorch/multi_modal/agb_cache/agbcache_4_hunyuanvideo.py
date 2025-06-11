# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, List, Tuple, Optional, Union, Dict
from dataclasses import dataclass

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens

from ascend_utils.common.security import get_valid_read_path, get_valid_write_path
from msmodelslim import logger


@dataclass
class AgbCacheInputsHunYuanVideo:
    x: torch.Tensor
    t: torch.Tensor
    text_states: Optional[torch.Tensor] = None
    text_mask: Optional[torch.Tensor] = None
    text_states_2: Optional[torch.Tensor] = None
    freqs_cos: Optional[torch.Tensor] = None
    freqs_sin: Optional[torch.Tensor] = None
    guidance: Optional[torch.Tensor] = None
    return_dict: bool = True


def agbcache_4_hunyuanvideo_forward(
    self,
    inputs: AgbCacheInputsHunYuanVideo
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x = inputs.x
        t = inputs.t
        text_states = inputs.text_states
        text_mask = inputs.text_mask
        text_states_2 = inputs.text_states_2
        freqs_cos = inputs.freqs_cos
        freqs_sin = inputs.freqs_sin
        guidance = inputs.guidance
        return_dict = inputs.return_dict
        
        out = {}

        if x is None:
            raise ValueError("Input 'x' is None, cannot proceed.")
        if text_states is None:
            raise ValueError("Input 'text_states' is None, cannot proceed.")
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        if t is None:
            raise ValueError("Input 't' is None, cannot proceed.")
        vec = self.time_in(t)

        # text modulation
        if text_states_2 is None:
            raise ValueError("Input 'text_states_2' is None, cannot proceed.")
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        if text_mask is None:
            raise ValueError("Input 'text_mask' is None, cannot proceed.")
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        
        if self.enable_agbcache:
            inp = img.clone()
            vec_ = vec.clone()
            txt_ = txt.clone()
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
            normed_inp = self.double_blocks[0].img_norm1(inp)
            modulated_inp = modulate(
                normed_inp, shift=img_mod1_shift, scale=img_mod1_scale
            )
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp - self.previous_modulated_input \
                                        ).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp  
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0          
        
        if self.enable_agbcache:
            if not should_calc:
                img += self.previous_residual
            else:
                ori_img = img.clone()
                # --------------------- Pass through DiT blocks ------------------------
                for _, block in enumerate(self.double_blocks):
                    double_block_args = [
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                    ]

                    img, txt = block(*double_block_args)

                # Merge txt and img to pass through single stream blocks.
                x = torch.cat((img, txt), 1)
                if len(self.single_blocks) > 0:
                    for _, block in enumerate(self.single_blocks):
                        single_block_args = [
                            x,
                            vec,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                        ]

                        x = block(*single_block_args)

                img = x[:, :img_seq_len, ...]
                self.previous_residual = img - ori_img
        else:        
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]

                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    ]

                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./hunyuanvideo")
    parser.add_argument('--save_path', type=str, default="./generated_videos")
    parser.add_argument('--enable_agbcache', type=bool, default=True)
    parser.add_argument('--infer_steps', type=int, default=50)
    parser.add_argument('--rel_l1_thresh', type=float, choices=[0.1, 0.15], default=0.15, \
                        help="0.1 for 1.6x speedup, 0.15 for 2.1x speedup")
    args = parser.parse_args()

    models_root_path = Path(args.model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    args = parse_args()

    # Create save folder to save the samples
    save_path = get_valid_write_path(args.save_path, write_mode=0o750)
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models and get the updated args
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args

    hunyuan_video_sampler.pipeline.transformer.__class__.enable_agbcache = args.enable_agbcache
    hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
    hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh = args.rel_l1_thresh
    hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_modulated_input = None
    hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = agbcache_4_hunyuanvideo_forward
    
    try:
        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        samples = outputs["samples"]
        
        if not outputs or "samples" not in outputs or not outputs["samples"]:
            raise ValueError("Pipeline did not return any valid samples.")
        else:
            samples = outputs["samples"]
            # Save samples
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                for i, sample in enumerate(samples):
                    sample = sample.unsqueeze(0)
                    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                    save_path = f"{save_path}/{time_flag}_seed{ \
                            outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                    save_path = get_valid_write_path(save_path, write_mode=0o750)
                    save_videos_grid(sample, save_path, fps=24)
                    logger.info(f'Sample save to: {save_path}')
                
    except Exception as e:
        raise ValueError(f"Error occurred during video generation or saving: {e}") from e
