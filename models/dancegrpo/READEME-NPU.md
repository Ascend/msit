# DanceGPRO—NPU适配
## 环境安装
### 依赖库安装
安装CANN、pytorch、torch_npu，pytorch推荐使用2.6.0版本。

### 迁移适配

- 解释器适配

  - 找到当前使用的python根目录下的lib/python3.10/site-packages/diffusers/models/embeddings.py文件，1250行:

    - 修改如下：

    - ```python
      is_mps = ids.device.type == "mps"
      is_npu = ids.device.type == "npu"
      freqs_dtype = torch.float32 if is_mps or is_npu else torch.float64
      ```

- 其余代码适配已在本仓代码中修改

## 性能优化

- 找到当前使用的python根目录下lib/site-packages/diffusers/models/embeddings.py/get_1d_rotary_pos_embed

  - 找到该行代码

    - ```python
      freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
      freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
      ```

  - 修改为

    - ```python
      freqs_cos = freqs.cos().T.repeat_interleave(2, dim=0).T.contiguous().float()
      freqs_sin = freqs.sin().T.repeat_interleave(2, dim=1).T.contiguous().float()
      ```

- 修改lib/site-packages/diffusers/models/attention_processor.py/Attention

  - ```python
    elif qk_norm == "rms_norm":
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        # 上面两行修改为
        self.norm_q = NpuFusedRMSNorm(dim_head, eps=eps)
        self.norm_k = NpuFusedRMSNorm(dim_head, eps=eps)
    ```

  - 增加类

    - ```python
      class NpuFusedRMSNorm(torch.nn.Module):
          def __init__(self, hidden_size, eps=1e-6):
              super().__init__()
              self.weight = nn.Parameter(torch.ones(hidden_size))
              self.eps = eps
      
          def forward(self, x):
              return torch_npu.npu_rms_norm(x.to(self.weight.dtype), self.weight, epsilon=self.eps)[0]
      ```

      

  