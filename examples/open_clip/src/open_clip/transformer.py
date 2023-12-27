import logging
import math
import numbers
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from .layer import MultiheadAttention
from .pos_embed import get_2d_sincos_pos_embed
from .utils import normal_, reset_parameters_torch, to_2tuple, zeros_

_logger = logging.getLogger(__name__)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, normalized_shape, epsilon=1e-5, dtype=ms.float32):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        super().__init__(normalized_shape, epsilon=epsilon, dtype=dtype)

    def construct(self, x: Tensor):
        orig_type = x.dtype
        x, _, _ = self.layer_norm(x.to(ms.float32), self.gamma.to(ms.float32), self.beta.to(ms.float32))
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def __init__(self, normalized_shape, epsilon=1e-5, dtype=ms.float32):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        super().__init__(normalized_shape, epsilon=epsilon, dtype=dtype)

    def construct(self, x: Tensor):
        orig_type = x.dtype
        x, _, _ = self.layer_norm(x, self.gamma.to(x.dtype), self.beta.to(x.dtype))
        return x.to(orig_type)


class GELU(nn.GELU):
    def __init__(self, approximate: str = "none"):
        if approximate == "none":
            super().__init__(False)
        elif approximate == "tanh":
            super().__init__(True)
        else:
            raise ValueError(f"approximate must be one of ['none', 'tanh'], but got {approximate}.")


class QuickGELU(nn.Cell):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        assert inplace is False, "Inplace is not supported for LayerScale."
        self.inplace = inplace
        self.gamma = Parameter(init_values * ops.ones(dim))

    def construct(self, x):
        return x * self.gamma


class PatchDropout(nn.Cell):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def construct(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]

        batch = x.shape[0]
        num_tokens = x.shape[1]

        batch_indices = ops.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = ops.randn((batch, num_tokens))
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1)[1]

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = ops.cat((cls_tokens, x), axis=1)

        return x


class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1.0 / 0.01),
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = Parameter(ops.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = Parameter(ops.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = Parameter(ops.log(10 * ops.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = Parameter(ops.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def construct(self, x, attn_mask: Optional[Tensor] = None):
        L, N, C = x.shape
        q, k, v = ops.dense(x, self.in_proj_weight, self.in_proj_bias).chunk(3, axis=-1)
        q = q.view((L, N * self.num_heads, -1)).swapaxes(0, 1)
        k = k.view((L, N * self.num_heads, -1)).swapaxes(0, 1)
        v = v.view((L, N * self.num_heads, -1)).swapaxes(0, 1)

        if self.logit_scale is not None:
            attn = ops.bmm(ops.L2Normalize(-1)(q), ops.L2Normalize(-1)(k).swapaxes(-1, -2))
            logit_scale = ops.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view((N, self.num_heads, L, L)) * logit_scale
            attn = attn.view((-1, L, L))
        else:
            q = q * self.scale
            attn = ops.bmm(q, k.swapaxes(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                new_attn_mask = ops.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask = new_attn_mask.masked_fill(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = ops.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view((N, self.num_heads, L, C)) * self.head_scale
            x = x.view((-1, L, C))
        x = x.swapaxes(0, 1).reshape((L, N, C))
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Cell):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = Parameter(ops.randn((n_queries, d_model)))
        self.attn = MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def construct(self, x: Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).tile(1, N, 1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class ResidualAttentionBlock(nn.Cell):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Dense(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
        self,
        q_x: Tensor,
        k_x: Optional[Tensor] = None,
        v_x: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def construct(
        self,
        q_x: Tensor,
        k_x: Optional[Tensor] = None,
        v_x: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Cell):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = GELU,
        norm_layer: Callable = LayerNorm,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("ln", norm_layer(mlp_width) if scale_fc else nn.Identity()),
                    ("c_proj", nn.Dense(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def construct(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token: Tensor, batch_size: int):
    return token.view(1, 1, -1).tile(batch_size, 1, 1)


class Transformer(nn.Cell):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.CellList(
            [
                ResidualAttentionBlock(
                    width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer
                )
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> ms.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, "int8_original_dtype"):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def construct(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Cell):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        attentional_pool: bool = False,
        attn_pooler_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.0,
        no_ln_pre: bool = False,
        pos_embed_type: str = "learnable",
        pool_type: str = "tok",
        final_ln_after_pool: bool = False,
        act_layer: Callable = GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ("tok", "avg", "none")
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, has_bias=False
        )

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.class_embedding = Parameter(scale * ops.randn(width))
        if pos_embed_type == "learnable":
            self.positional_embedding = Parameter(scale * ops.randn((self.grid_size[0] * self.grid_size[1] + 1, width)))
        elif pos_embed_type == "sin_cos_2d":
            # fixed sin-cos embedding
            assert (
                self.grid_size[0] == self.grid_size[1]
            ), "currently sin cos 2d pos embedding only supports square input"
            self.positional_embedding = Parameter(
                ops.zeros((self.grid_size[0] * self.grid_size[1] + 1, width)), requires_grad=False
            )
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.set_data(Tensor(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = "none"
                if attentional_pool in ("parallel", "cascade"):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ""
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = Parameter(scale * ops.randn((pool_dim, output_dim)))

        self.apply(reset_parameters_torch)
        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.get_parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.get_parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default init, below, or alternate init is best.

        pass

    def set_grad_checkpointing(self, enable=True):
        _logger.warning("Gradient checkpointing is not supported. Setting enable is omitted.")
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.pool_type == "avg":
            pooled, tokens = x[:, 1:].mean(axis=1), x[:, 1:]
        elif self.pool_type == "tok":
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled, tokens = x, x

        return pooled, tokens

    def construct(self, x: Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape((x.shape[0], x.shape[1], -1))  # shape = [*, width, grid ** 2]
        x = x.permute((0, 2, 1))  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = ops.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], axis=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.permute((1, 0, 2))  # LND -> NLD

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == "parallel":
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == "cascade"
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled


def text_global_pool(x, text: Optional[Tensor] = None, pool_type: str = "argmax"):
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[ops.arange(x.shape[0]), text.argmax(axis=-1)], x
    else:
        pooled, tokens = x, x

    return pooled, tokens


class TextTransformer(nn.Cell):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        output_dim: int = 512,
        embed_cls: bool = False,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = Parameter(ops.zeros(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = Parameter(ops.zeros((self.num_pos, width)))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.attn_mask = self.build_causal_mask()

        self.proj_bias = proj_bias
        if proj_bias:
            self.text_projection = nn.Dense(width, output_dim)
        else:
            self.text_projection = Parameter(ops.zeros((width, output_dim)))

        self.apply(reset_parameters_torch)
        self.init_parameters()

    def init_parameters(self):
        normal_(self.token_embedding.embedding_table, std=0.02)
        normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            normal_(block.attn.in_proj_weight, std=attn_std)
            normal_(block.attn.out_proj.weight, std=proj_std)
            normal_(block.mlp.c_fc.weight, std=fc_std)
            normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Dense):
                normal_(self.text_projection.weight, std=self.transformer.width**-0.5)
                if self.text_projection.bias is not None:
                    zeros_(self.text_projection.bias)
            else:
                normal_(self.text_projection, std=self.transformer.width**-0.5)

    def set_grad_checkpointing(self, enable=True):
        _logger.warning("Gradient checkpointing is not supported. Setting enable is omitted.")
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = ops.zeros((self.num_pos, self.num_pos))
        mask = mask.fill(float("-inf"))
        mask = mask.triu(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: ms.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = ops.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = ops.zeros_like(cls_mask.shape, dtype=cast_dtype)
        additive_mask = additive_mask.masked_fill(~cls_mask, float("-inf"))
        additive_mask = additive_mask.repeat_interleave(self.heads, 0)
        return additive_mask

    def construct(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = ops.cat([x, _expand_token(self.cls_emb, x.shape[0]).to(cast_dtype)], axis=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute((1, 0, 2))  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute((1, 0, 2))  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type="last")
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if self.proj_bias:
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        context_length: int = 77,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = GELU,
        norm_layer: Callable = LayerNorm,
        output_dim: int = 512,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.CellList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    is_cross_attention=True,
                )
                for _ in range(layers)
            ]
        )

        self.attn_mask = self.build_attention_mask()
        self.ln_final = norm_layer(width)
        self.text_projection = Parameter(ops.zeros((width, output_dim)))

        self.apply(reset_parameters_torch)
        self.init_parameters()

    def init_parameters(self):
        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            normal_(block.attn.in_proj_weight, std=attn_std)
            normal_(block.attn.out_proj.weight, std=proj_std)
            normal_(block.mlp.c_fc.weight, std=fc_std)
            normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            normal_(block.attn.in_proj_weight, std=attn_std)
            normal_(block.attn.out_proj.weight, std=proj_std)
            normal_(block.mlp.c_fc.weight, std=fc_std)
            normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        mask = ops.fill(ms.float32, (self.context_length, self.context_length), float("-inf"))
        mask = mask.triu(1)  # zero out the lower diagonal
        return mask

    def construct(self, image_embs, text_embs):
        text_embs = text_embs.permute((1, 0, 2))  # NLD -> LNDsq
        image_embs = image_embs.permute((1, 0, 2))  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
            text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute((1, 0, 2))  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    def set_grad_checkpointing(self, enable=True):
        _logger.warning("Gradient checkpointing is not supported. Setting enable is omitted.")
        self.grad_checkpointing = enable
