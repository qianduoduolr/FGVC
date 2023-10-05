import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ....registry import COMPONENTS
from mmcv.runner import BaseModule
from timm.models.vision_transformer import Attention, Mlp



class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU() # need to fix approximate="tanh"
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

@COMPONENTS.register_module()
class SpaTempFormer(BaseModule):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=12,
        time_depth=12,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        update_feat=True,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.update_feat = update_feat
        self.out_channels = 2
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, input_tensor, other=None):
        
        x = self.input_transform(input_tensor.permute(0, 2, 1, 3))

        j = 0
        for i in range(len(self.time_blocks)):
            B, N, T, _ = x.shape
            x_time = rearrange(x, "b n t c -> (b n) t c", b=B, t=T, n=N)
            x_time = self.time_blocks[i](x_time)

            x = rearrange(x_time, "(b n) t c -> b n t c ", b=B, t=T, n=N)
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_blocks)) == 0
            ):
                x_space = rearrange(x, "b n t c -> (b t) n c ", b=B, t=T, n=N)
                x_space = self.space_blocks[j](x_space)
                x = rearrange(x_space, "(b t) n c -> b n t c  ", b=B, t=T, n=N)
                j += 1

        flow = self.flow_head(x).permute(0, 2, 1, 3)
        
        
        return flow
    
    

@COMPONENTS.register_module()
class ContextSpaTempFormer(BaseModule):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=12,
        time_depth=12,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        update_feat=True,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.update_feat = update_feat
        self.out_channels = 2
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform_time = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.input_transform_spa = torch.nn.Linear(input_dim, hidden_size, bias=True)
        
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x_time, x_space):
        
        x_time = self.input_transform_time(x_time.permute(0, 2, 1, 3))
        x_space = self.input_transform_spa(x_space.permute(0, 2, 1, 3))
        
        j = 0
        for i in range(len(self.time_blocks)):
            B, N, T, _ = x_time.shape
            x_time = rearrange(x_time, "b n t c -> (b n) t c", b=B, t=T, n=N)
            x_time = self.time_blocks[i](x_time)

            x_time = rearrange(x_time, "(b n) t c -> b n t c ", b=B, t=T, n=N)
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_blocks)) == 0
            ):
                x_space = rearrange(x_space, "b n t c -> (b t) n c ", b=B, t=T, n=N)
                x_space = self.space_blocks[j](x_space)
                x_space = rearrange(x_space, "(b t) n c -> b n t c  ", b=B, t=T, n=N)
                j += 1
                
            context_att = torch.einsum("")
            x_time = x_time * context_att

        flow = self.flow_head(x_time).permute(0, 2, 1, 3)
        
        
        return flow