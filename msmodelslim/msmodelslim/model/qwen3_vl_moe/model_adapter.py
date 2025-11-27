#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Generator, Tuple, Dict
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextDecoderLayer,
    Qwen3VLMoeTextSparseMoeBlock
)

from msmodelslim.app import DeviceType
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.factory import ModelFactory
from msmodelslim.model.interface_hub import (
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    ModelSlimPipelineInterfaceV1
)
from msmodelslim.model.vlm_base import VLMBaseModelAdapter
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_32G

from .moe_utils import UnstackedQwen3VLMoeSparseMoeBlock


@ModelFactory.register("Qwen3-VL-30B-A3B")
@ModelFactory.register("Qwen3-VL-235B-A22B")
@logger_setter()
class Qwen3VLMoeModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    IterSmoothInterface,
    FlexSmoothQuantInterface
):
    """
    V1 Framework adapter for Qwen3-VL-MoE models.
    
    Key features:
    - Layer-wise loading for text decoder
    - Vision encoder processed as a whole
    - Automatic MoE fusion layer conversion via MoeConverterProcessor
    - Multimodal calibration dataset support
    
    Architecture:
        model.visual (VisionEncoder) - Loaded once, processed first
        model.language_model.layers[i] (TextDecoder) - Loaded layer-by-layer
    """
    
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        # Cache for processor (used in dataset handling)
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
        
        # Initialize attention heads config (required for OV smoothing)
        self.num_attention_heads, self.num_key_value_heads = self._init_num_attention_heads()
    
    def get_model_pedigree(self) -> str:
        """Return model pedigree for best practice matching"""
        return 'qwen3_vl_moe'
    
    def get_model_type(self) -> str:
        """Return model type"""
        return self.model_type
    
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        Handle multimodal visual language model calibration dataset.
        
        Expected dataset format (list of dict):
            [
                {
                    'image': '<path_to_image>',
                    'text': '<prompt_text>'  # optional, defaults to "Describe this image."
                },
                ...
            ]
        
        Returns:
            List of preprocessed inputs ready for model forward:
            [
                {
                    'input_ids': Tensor,
                    'attention_mask': Tensor,
                    'pixel_values': Tensor,
                    'image_grid_thw': Tensor
                },
                ...
            ]
        """
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True
        )
        
        # Preprocess each sample
        processed_data = []
        for item in tqdm(dataset, desc="Processing calibration dataset"):
            image_path = item.get('image')
            text = item.get('text', 'Describe this image.')
            
            # Validate image path
            image_path = get_valid_read_path(image_path)
            
            # Apply chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": text}
                    ]
                }
            ]
            
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to target device and save
            processed_data.append({
                'input_ids': inputs.input_ids.to(device.value),
                'attention_mask': inputs.attention_mask.to(device.value),
                'position_ids': None,
                'past_key_values': None,
                'inputs_embeds': None,
                'labels': None,
                'pixel_values': inputs.pixel_values.to(device.value),
                'pixel_values_videos': None,
                'image_grid_thw': inputs.image_grid_thw.to(device.value),
                'video_grid_thw': None,
                'cache_position': None,
                'logits_to_keep': 0
            })
        
        get_logger().info(f"Processed {len(processed_data)} multimodal vlm samples")
        return processed_data
    
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Initialize model with vision encoder on CPU and text decoder with only 1 layer.
        
        Strategy (similar to DeepSeek-V3):
            - Save original layer count
            - Temporarily set num_hidden_layers to 1
            - Load model with vision encoder + 1 text decoder layer
            - Restore original layer count
            - Other layers will be loaded on-demand in generate_decoder_layer
        
        Returns:
            Model with vision encoder + 1 decoder layer loaded, others on meta
        """
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
        except ImportError as e:
            raise InvalidModelError(
                "Failed to import Qwen3VLMoeForConditionalGeneration. "
                "Please install transformers with Qwen3-VL-MoE support.",
                action="pip install transformers==4.57.1"
            ) from e
        
        get_logger().info("Initializing Qwen3-VL-MoE model with v1 framework (layer-wise loading)...")
        
        # Save original layer count
        origin_layers = self.config.text_config.num_hidden_layers
        get_logger().info(f"Model with {origin_layers} text layers + {self.config.vision_config.depth} vision layers")
        
        # Temporarily set to 1 layer for initialization
        self.config.text_config.num_hidden_layers = 1
        self.config.use_cache = False  # Disable cache to save memory
        
        # Validate model path
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)
        
        # Load model with only 1 text decoder layer
        # Vision encoder is fully loaded, text decoder has only 1 layer
        get_logger().info("Loading vision encoder and first text decoder layer...")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            torch_dtype="auto",
            local_files_only=True,
            device_map="cpu",  # All on CPU for now
            attn_implementation='eager'  # Required: prevents KeyError when accessing ALL_ATTENTION_FUNCTIONS
        ).eval()
        
        # Restore original layer count
        self.config.text_config.num_hidden_layers = origin_layers
        
        # Ensure _attn_implementation is set for dynamically loaded layers
        # This prevents KeyError when layers access ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
        self.config.text_config._attn_implementation = 'eager'
        
        # Load full state_dict for the first layer + vision encoder + lm_head
        get_logger().info("Loading weights for vision encoder, first decoder layer, and lm_head...")
        state_dict = self._get_state_dict(model)
        model.load_state_dict(state_dict)
        
        # CRITICAL: Copy text_config attention heads to model.config for OV smoothing
        # BaseSmoothProcessor._apply_standard_ov_smooth() reads from model.config, not model.config.text_config
        # This must be done AFTER model is loaded
        if hasattr(model.config.text_config, 'num_attention_heads'):
            model.config.num_attention_heads = model.config.text_config.num_attention_heads
            get_logger().info(f"Set model.config.num_attention_heads = {model.config.num_attention_heads}")
        if hasattr(model.config.text_config, 'num_key_value_heads'):
            model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
            get_logger().info(f"Set model.config.num_key_value_heads = {model.config.num_key_value_heads}")
        
        get_logger().info(f"Model initialized with {origin_layers} layers (1 loaded, others will be loaded on-demand)")
        
        # IMPORTANT: Convert layer 0 if it's a MoE layer
        # Layer 0 is loaded in init_model, but other layers are loaded in _load_decoder_if_not_exist
        # So we need to explicitly convert layer 0 here
        if self._is_moe_layer(0):
            get_logger().info("Layer 0 is a MoE layer, performing architecture adaptation...")
            decoder_layer_0 = model.model.language_model.layers[0]
            self._convert_single_moe_layer(decoder_layer_0, 0)
            get_logger().info("Layer 0 architecture adaptation completed")
        
        return model
    
    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model visit pipeline for layer-wise processing.
        
        Uses the common layer-wise visit function for consistent behavior.
        
        Processing order:
            1. Vision encoder (model.visual) - processed as a whole
            2. Text decoder layers (model.language_model.layers[0..N]) - loaded on-demand
        
        Yields:
            ProcessRequest(name, module, args, kwargs)
        """
        # 1. Process vision encoder first
        get_logger().info("Processing vision encoder...")
        yield ProcessRequest(
            name="model.visual",
            module=model.model.visual,
            args=(),
            kwargs={}
        )
        
        # 2. Process text decoder layers one by one using standard visit function
        get_logger().info("Processing text decoder layers...")
        yield from generated_decoder_layer_visit_func(
            model, 
            transformer_blocks=self.generate_decoder_layer(model)
        )
    
    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Generate model forward pipeline for calibration.
        
        This is more complex as we need to:
            1. Run vision encoder to get image features
            2. Merge image features into text embeddings
            3. Run each text decoder layer with proper inputs
        
        Args:
            model: The model
            inputs: Preprocessed data from handle_dataset
        
        Yields:
            ProcessRequest with forward results
        """
        # For multimodal models, forward is more complex
        # We need to handle the vision-language fusion
        
        # 1. Extract first sample for calibration
        if isinstance(inputs, list):
            sample = inputs[0]
        else:
            sample = inputs
        
        # 2. Vision encoder forward
        pixel_values = sample['pixel_values']
        image_grid_thw = sample['image_grid_thw']
        
        with torch.no_grad():
            # Run vision encoder
            image_embeds, deepstack_image_embeds = model.model.visual(
                pixel_values, grid_thw=image_grid_thw
            )
        
        # Yield vision encoder result
        yield ProcessRequest(
            name="model.visual",
            module=model.model.visual,
            args=(pixel_values, image_grid_thw),
            kwargs={}
        )
        
        # 3. Prepare inputs for text decoder
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        
        # Get input embeddings and merge with image features
        inputs_embeds = model.model.language_model.embed_tokens(input_ids)
        
        # Get cache_position for attention mask creation
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        
        # Get position ids
        position_ids, rope_deltas = model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask
        )
        
        # Expand position_ids if needed (3D format for mROPE)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        # Extract text position ids
        text_position_ids = position_ids[0]
        
        # CRITICAL: Convert 2D attention_mask to 4D causal mask
        # This is what Qwen3VLMoeTextModel.forward does internally
        attention_mask = create_causal_mask(
            config=model.config.text_config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        
        # Create position embeddings (shared across layers)
        position_embeddings = model.model.language_model.rotary_emb(inputs_embeds, position_ids)
        
        # 4. Process each decoder layer
        hidden_states = inputs_embeds
        for name, layer in self.generate_decoder_layer(model):
            with torch.no_grad():
                # Forward through current layer
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,  # Now 4D: [batch, 1, seq_len, seq_len]
                    position_ids=text_position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    past_key_values=None,
                    use_cache=False,
                )
            
            # Yield layer result
            yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    'attention_mask': attention_mask,
                    'position_ids': text_position_ids,
                    'cache_position': cache_position,
                    'position_embeddings': position_embeddings,
                    'past_key_values': None,
                    'use_cache': False,
                }
            )
    
    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Generate decoder layers, loading them on-demand.
        
        Similar to DeepSeekV3's approach but for Qwen3-VL-MoE.
        Each layer is loaded from safetensors file, and MoE layers are converted immediately.
        
        Yields:
            (layer_name, layer_module) tuples
        """
        num_layers = self.config.text_config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            name = f"model.language_model.layers.{layer_idx}"
            
            # Load layer if not exists (includes MoE conversion for MoE layers)
            layer = self._load_decoder_if_not_exist(model, name, layer_idx)
            
            yield name, layer
    
    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable/disable KV cache.
        
        For calibration, we typically don't need KV cache.
        """
        model.config.use_cache = need_kv_cache
        get_logger().info(f"KV cache {'enabled' if need_kv_cache else 'disabled'}")
    
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Get adapter config for subgraph-based anti-outlier processing (iter_smooth).
        
        Defines the subgraph structure for norm-linear, ov, and other fusions.
        Based on qwen3vl.py implementation but adapted for Qwen3-VL-MoE model.
        
        Includes both vision encoder and text decoder layers.
        """
        adapter_config = []
        
        # Text decoder layers
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            # Norm-Linear: input_layernorm -> QKV
            norm_linear_mapping_config = MappingConfig(
                source=f"model.language_model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.language_model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.language_model.layers.{layer_idx}.self_attn.v_proj"
                ]
            )
            
            # OV fusion: V -> O
            ov_mapping_config = MappingConfig(
                source=f"model.language_model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"model.language_model.layers.{layer_idx}.self_attn.o_proj"]
            )
            
            adapter_config.extend([
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=norm_linear_mapping_config
                ),
                AdapterConfig(
                    subgraph_type="ov",
                    mapping=ov_mapping_config,
                    extra_config={
                        'group_method': 'max',
                        'num_attention_heads': self.config.text_config.num_attention_heads,
                        'num_key_value_heads': self.config.text_config.num_key_value_heads
                    }
                ),
            ])
            
            if layer_idx not in self.config.text_config.mlp_only_layers:
                if (layer_idx + 1) % self.config.text_config.decoder_sparse_step != 0:
                    # Regular MLP layer
                    mlp_mapping_config = MappingConfig(
                        source=f"model.language_model.layers.{layer_idx}.post_attention_layernorm",
                        targets=[
                            f"model.language_model.layers.{layer_idx}.mlp.gate_proj",
                            f"model.language_model.layers.{layer_idx}.mlp.up_proj"
                        ]
                    )
                    up_down_mapping = MappingConfig(
                        source=f"model.language_model.layers.{layer_idx}.mlp.up_proj",
                        targets=[f"model.language_model.layers.{layer_idx}.mlp.down_proj"]
                    )
                    adapter_config.extend([
                        AdapterConfig(
                            subgraph_type="norm-linear",
                            mapping=mlp_mapping_config
                        ),
                        AdapterConfig(
                            subgraph_type="up-down",
                            mapping=up_down_mapping
                        )
                    ])
        
        return adapter_config
    
    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Get weight map from model.safetensors.index.json"""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']
    
    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """
        Load state dict for a specific module from safetensors files.
        
        Args:
            module: The module to load weights for
            prefix: Name prefix for the module in the full model
        
        Returns:
            State dict for the module
        """
        weight_map = self._get_weight_map()
        
        # Get all parameter names for this module
        param_names = [name for name, _ in module.named_parameters()]
        
        # Group by safetensors file
        file_groups = defaultdict(list)
        for param_name in param_names:
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_name = weight_map[full_name]
                file_groups[file_name].append(param_name)
        
        # Load weights file by file
        state_dict = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_32G)
            
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = f.get_tensor(full_name)
        
        return state_dict
    
    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        """
        Load a specific decoder layer from safetensors if not already loaded.
        
        This method:
        1. Checks if layer already exists and is loaded
        2. If not, creates layer structure (without initializing weights)
        3. Loads weights from safetensors files
        4. If it's a MoE layer, converts 3D fused weights to standard nn.Linear
        5. Returns the loaded (and potentially converted) layer
        
        Args:
            model: The model
            name: Full layer name (e.g., "model.language_model.layers.0")
            idx: Layer index
        
        Returns:
            Loaded decoder layer module
        """
        try:
            # Try to access the layer
            decoder = model.get_submodule(name)
            # Check if it's actually loaded (not on meta device)
            try:
                _ = decoder.input_layernorm.weight.device
                # If we can access the device, layer is loaded
                get_logger().debug(f"Layer {idx} already loaded")
                return decoder
            except RuntimeError:
                # Weight is on meta device, need to load
                pass
        except AttributeError:
            # Layer doesn't exist in the module list yet
            pass
        
        get_logger().info(f"Loading decoder layer {idx}...")
        
        # Disable reset_parameters to avoid slow and unnecessary initialization
        # We will load weights from safetensors immediately after
        with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
            get_logger().info(f'Creating decoder layer {idx} structure...')
            
            # Create layer structure (weights will be on meta or uninitialized)
            decoder = Qwen3VLMoeTextDecoderLayer(
                self.config.text_config,
                layer_idx=idx
            )
            
            # Load weights from safetensors
            state_dict = self._get_state_dict(decoder, prefix=name)
            decoder.load_state_dict(state_dict)
            decoder.eval()
            
            # Add layer to model's layer list
            module_list: nn.ModuleList = model.model.language_model.layers
            if len(module_list) <= idx:
                module_list.append(decoder)
            else:
                module_list[idx] = decoder
            
            get_logger().info(f'Decoder layer {idx} loaded successfully')
        
        # Perform architecture adaptation if needed
        # MoE conversion is part of model architecture adaptation, not quantization strategy
        # Similar to DeepSeek-V3's MTP layer wrapping in load_mtp_if_not_load
        if self._is_moe_layer(idx):
            get_logger().info(f"Layer {idx} is a MoE layer, performing architecture adaptation...")
            self._convert_single_moe_layer(decoder, idx)
            get_logger().info(f"Layer {idx} architecture adaptation completed")
        
        return decoder
    
    def _is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a given layer index is a MoE layer"""
        if layer_idx in self.config.text_config.mlp_only_layers:
            return False
        if (layer_idx + 1) % self.config.text_config.decoder_sparse_step == 0:
            return True
        return False
    
    def _convert_single_moe_layer(self, layer: nn.Module, layer_idx: int):
        """
        Convert a single MoE layer's 3D fused weights to standard nn.Linear layers.
        
        This is called per-layer during layer-wise loading, which is more memory-efficient
        than converting all MoE layers at once.
        
        Args:
            layer: The decoder layer module
            layer_idx: Layer index (for logging)
        """
        original_moe_block = layer.mlp
        
        # Verify it's actually a MoE block
        if not isinstance(original_moe_block, Qwen3VLMoeTextSparseMoeBlock):
            get_logger().warning(
                f"Layer {layer_idx} MLP is not a Qwen3VLMoeTextSparseMoeBlock, skipping conversion. "
                f"Got: {type(original_moe_block)}"
            )
            return
        
        # Create unstacked MoE block
        unstacked_moe_block = UnstackedQwen3VLMoeSparseMoeBlock(
            self.config.text_config,
            original_moe_block,
            copy_weights=False
        )
        
        # Transform weights from 3D fused to individual Linear layers
        unstacked_moe_block._transform_weights_from_original(
            original_moe_block,
            in_place=True  # Save memory by freeing original weights immediately
        )
        
        # Set to eval mode (critical: new module defaults to training mode)
        unstacked_moe_block.eval()
        
        # Replace the MLP in the layer
        layer.mlp = unstacked_moe_block
        
        # Clean up
        del original_moe_block
        gc.collect()

    def _init_num_attention_heads(self):
        """
        Initialize attention heads configuration.
        
        Required for OV smoothing and other attention-related processing.
        Based on qwen3vl.py implementation.
        
        Returns:
            Tuple of (num_attention_heads, num_key_value_heads)
        """
        num_attention_heads = None
        num_key_value_heads = None

        attention_heads_keys = ["num_attention_heads", "n_head", "num_heads"]
        key_value_heads_keys = ["num_key_value_heads"]

        # Check in text_config (Qwen3VLMoe has separate text and vision configs)
        for key in attention_heads_keys:
            if hasattr(self.config.text_config, key):
                num_attention_heads = getattr(self.config.text_config, key)
                break

        for key in key_value_heads_keys:
            if hasattr(self.config.text_config, key):
                num_key_value_heads = getattr(self.config.text_config, key)
                break

        if not num_attention_heads:
            raise ValueError(
                "the config of model must have num_attention_heads, n_head or num_heads, "
                "please check or modify the config file"
            )
        
        return num_attention_heads, num_key_value_heads