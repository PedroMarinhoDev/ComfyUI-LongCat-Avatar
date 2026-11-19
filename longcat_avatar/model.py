"""LongCat-Video-Avatar 1.6 DiT (meituan-longcat/LongCat-Video, longcat_video/modules/avatar/longcat_video_dit_avatar.py).

A 48-block, 4096-wide single-stream video DiT over Wan 2.1 latents, with UMT5 text cross-attention and a
per-frame Whisper audio cross-attention in every block. Module names follow the release so its checkpoint and
LoRAs load unchanged.

Token layout along time: [reference frame][conditioning frames][noisy frames]. The reference frame (later video
segments only) is the first segment's image latent placed at RoPE time ``ref_index``. Reference and
conditioning frames are clean (timestep 0) and attend only within their own group; noisy frames attend to
everything and are the only ones that read text and audio.

So the clean tokens are the same at every step of a sampling call. Given a ``CleanKV`` in transformer_options
(``longcat_clean_kv``, made per sampling call by DistillGuider), the first step keeps their per-block attention keys
and values and their output, and later steps run only the noisy tokens.
"""
import torch
import torch.nn.functional as F
from torch import nn

import comfy.ldm.common_dit
import comfy.ldm.flux.math
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.wan.model_multitalk import MultiTalkAudioProjModel

FFN_CHUNK = 8192


class CleanKV:
    """The clean tokens' per-block keys and values and final output, for one sampling call of one set of weights,
    keyed by the batch's cond uuids."""
    def __init__(self):
        self.entries = {}


def modulated_norm(x, norm, scale, shift):
    return comfy.quant_ops.ck.adaln(x, scale, shift, norm.eps)


def rms_rope(q, k, q_norm, k_norm, pe):
    q_scale, _, q_stream = comfy.ops.cast_bias_weight(q_norm, q, offloadable=True)
    k_scale, _, k_stream = comfy.ops.cast_bias_weight(k_norm, k, offloadable=True)
    q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, q_norm.eps)
    comfy.ops.uncast_bias_weight(q_norm, q_scale, None, q_stream)
    comfy.ops.uncast_bias_weight(k_norm, k_scale, None, k_stream)
    return q, k


def attention(q, k, v, heads, comfy_attention, transformer_options):
    q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
    return optimized_attention(q, k, v, heads, transformer_options=transformer_options, preferred_attention=comfy_attention)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=True, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, pe, groups, clean_kv, transformer_options):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.num_heads, -1).unbind(2)
        q, k = rms_rope(q, k, self.q_norm, self.k_norm, pe)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        # clean_kv: None, [] to keep the clean tokens' keys and values (the noisy group starts after them), or [k, v] to prepend
        if clean_kv:
            k, v = torch.cat([clean_kv[0], k], dim=1), torch.cat([clean_kv[1], v], dim=1)
        elif clean_kv is not None:
            n = groups[-1][0]
            with comfy.model_prefetch.pause_malloc_graph():
                clean_kv += [k[:, :n].clone(), v[:, :n].clone()]
        # groups: (query start, query end, key end); clean frames see only their own group, noisy frames see all
        out = [attention(q[:, qs:qe], k[:, ks:ke], v[:, ks:ke], self.num_heads, self.comfy_attention, transformer_options)
               for qs, qe, ks, ke in groups]
        return self.proj(out[0] if len(out) == 1 else torch.cat(out, dim=1))


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.kv_linear = operations.Linear(context_dim, dim * 2, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, context, transformer_options):
        B, N, _ = x.shape
        q = self.q_norm(self.q_linear(x).view(B, N, self.num_heads, -1)).flatten(2)
        k, v = self.kv_linear(context).view(B, context.shape[1], 2, self.num_heads, -1).unbind(2)
        k = self.k_norm(k).flatten(2)
        return self.proj(attention(q, k, v.flatten(2), self.num_heads, self.comfy_attention, transformer_options))


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AvatarBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, adaln_dim, audio_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(adaln_dim, 6 * dim, dtype=dtype, device=device))
        self.audio_adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(adaln_dim, 3 * dim, dtype=dtype, device=device))
        self.mod_norm_attn = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.mod_norm_ffn = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.pre_crs_attn_norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.pre_video_crs_attn_norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.attn = SelfAttention(dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.cross_attn = CrossAttention(dim, dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.audio_cross_attn = CrossAttention(dim, audio_dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.ffn = FeedForwardSwiGLU(dim, ffn_dim, dtype=dtype, device=device, operations=operations)

    def forward(self, x, context, audio, t, pe, groups, num_clean, clean_kv, transformer_options={}):
        """x (B, T*S, C); t (B, T, adaln_dim); audio (B * noisy frames, 32, audio_dim); the first num_clean frames are clean."""
        B, N, C = x.shape
        T = t.shape[1]
        S = N // T
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)

        xt = x.view(B, T, S, C)
        y = self.attn(modulated_norm(xt, self.mod_norm_attn, scale_msa, shift_msa).view(B, N, C), pe, groups, clean_kv, transformer_options)
        xt.addcmul_(y.view(B, T, S, C), gate_msa)
        del y

        # text and audio only reach the noisy frames
        xn = x[:, num_clean * S:]
        xn.add_(self.cross_attn(self.pre_crs_attn_norm(xn), context, transformer_options))
        frames = T - num_clean
        y = self.audio_cross_attn(self.pre_video_crs_attn_norm(xn).view(B * frames, S, C), audio, transformer_options)
        audio_shift, audio_scale, audio_gate = self.audio_adaLN_modulation(t[:, num_clean:]).unsqueeze(2).chunk(3, dim=-1)
        xn.view(B, frames, S, C).addcmul_(modulated_norm(y.view(B, frames, S, C), self.mod_norm_attn, audio_scale, audio_shift), audio_gate)
        del y, xn

        # frame chunks bound the SwiGLU intermediate; the FFN is per token, so the result is unchanged
        step = max(1, FFN_CHUNK // S)
        for f in range(0, T, step):
            chunk = xt[:, f:f + step]
            chunk.addcmul_(self.ffn(modulated_norm(chunk, self.mod_norm_ffn, scale_mlp[:, f:f + step], shift_mlp[:, f:f + step])), gate_mlp[:, f:f + step])
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_channels, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size, dtype=dtype, device=device)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, frequency_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(operations.Linear(frequency_dim, dim, dtype=dtype, device=device), nn.SiLU(),
                                 operations.Linear(dim, dim, dtype=dtype, device=device))

    def forward(self, t, dtype):
        # the release runs this and the modulations under fp32 autocast; here they run in the model dtype, like Wan
        return self.mlp(timestep_embedding(t.float(), self.frequency_dim, time_factor=1.0).to(dtype))


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.y_proj = nn.Sequential(operations.Linear(in_channels, dim, dtype=dtype, device=device), nn.GELU(approximate="tanh"),
                                    operations.Linear(dim, dim, dtype=dtype, device=device))


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim, adaln_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.linear = operations.Linear(dim, out_dim, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(adaln_dim, 2 * dim, dtype=dtype, device=device))

    def forward(self, x, t):
        B, N, C = x.shape
        T = t.shape[1]
        shift, scale = self.adaLN_modulation(t).unsqueeze(2).chunk(2, dim=-1)
        return self.linear(modulated_norm(x.view(B, T, -1, C), self.norm_final, scale, shift).view(B, N, C))


class LongCatVideoAvatar(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, hidden_size=4096, depth=48, num_heads=32, caption_channels=4096,
                 mlp_ratio=4, adaln_tembed_dim=512, frequency_embedding_size=256, patch_size=(1, 2, 2), audio_window=5,
                 audio_block=5, audio_channel=1280, intermediate_dim=512, output_dim=768, context_tokens=32, vae_scale=4,
                 image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.out_channels = out_channels
        self.patch_size = tuple(patch_size)
        self.num_heads = num_heads
        self.vae_scale = vae_scale
        self.audio_window = audio_window
        head_dim = hidden_size // num_heads
        # head_dim 128 -> 44 time, 42 height and 42 width channels
        self.rope_axes_dim = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
        # FeedForwardSwiGLU sizing: 2/3 of mlp_ratio * dim, rounded up to a multiple of 256
        ffn_dim = 256 * ((int(2 * hidden_size * mlp_ratio / 3) + 255) // 256)

        self.x_embedder = PatchEmbed3D(self.patch_size, in_channels, hidden_size, dtype=dtype, device=device, operations=operations)
        self.t_embedder = TimestepEmbedder(adaln_tembed_dim, frequency_embedding_size, dtype=dtype, device=device, operations=operations)
        self.y_embedder = CaptionEmbedder(caption_channels, hidden_size, dtype=dtype, device=device, operations=operations)
        self.blocks = nn.ModuleList([AvatarBlock(hidden_size, num_heads, ffn_dim, adaln_tembed_dim, output_dim, dtype=dtype, device=device, operations=operations)
                                     for _ in range(depth)])
        self.audio_proj = MultiTalkAudioProjModel(seq_len=audio_window, seq_len_vf=audio_window + vae_scale - 1, blocks=audio_block,
                                                  channels=audio_channel, intermediate_dim=intermediate_dim, out_dim=output_dim,
                                                  context_tokens=context_tokens, dtype=dtype, device=device, operations=operations)
        self.final_layer = FinalLayer(hidden_size, out_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                                      adaln_tembed_dim, dtype=dtype, device=device, operations=operations)

    def project_audio(self, audio):
        """(B, frames, window, layers, C) Whisper windows -> (B, latent frames, 32, output_dim).

        The first frame keeps its own window; each later group of vae_scale frames is folded into one latent frame
        from the first frame's left half-window, the middle frames' centres and the last frame's right half-window.
        """
        mid = self.audio_window // 2
        first = audio[:, :1]
        later = audio[:, 1:].unflatten(1, (-1, self.vae_scale))
        later = torch.cat([later[:, :, :1, :mid + 1].flatten(2, 3),
                           later[:, :, 1:-1, mid:mid + 1].flatten(2, 3),
                           later[:, :, -1:, mid:].flatten(2, 3)], dim=2)
        return self.audio_proj(first, later)

    def rope(self, t, h, w, ref_index, device):
        frames = torch.arange(t, device=device, dtype=torch.float32)
        if ref_index is not None:
            # the reference frame sits at time ref_index, the video restarts at 0 after it
            frames = torch.cat([frames.new_full((1,), ref_index), frames[:-1]])
        ids = torch.stack(torch.meshgrid(frames, torch.arange(h, device=device, dtype=torch.float32),
                                         torch.arange(w, device=device, dtype=torch.float32), indexing="ij"), dim=-1).reshape(1, -1, 3)
        pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000) for i, d in enumerate(self.rope_axes_dim)], dim=-3)
        return pe.unsqueeze(2)

    def forward(self, x, timestep, context, audio_embeds, cond_latent=None, ref_latent=None, ref_index=10, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, audio_embeds, cond_latent, ref_latent, ref_index, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, audio_embeds, cond_latent=None, ref_latent=None, ref_index=10, transformer_options={}, **kwargs):
        B, _, T, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()

        num_cond_latents = 0
        timestep = timestep.reshape(B, 1).expand(B, T)
        if cond_latent is not None:
            # the conditioning frames replace the first frames, clean, at timestep 0
            num_cond_latents = cond_latent.shape[2]
            x = torch.cat([cond_latent, x[:, :, num_cond_latents:]], dim=2)
            timestep = torch.cat([timestep.new_zeros((B, num_cond_latents)), timestep[:, num_cond_latents:]], dim=1)
        num_ref = 0
        if ref_latent is not None:
            num_ref = 1
            x = torch.cat([ref_latent, x], dim=2)
            timestep = torch.cat([timestep.new_zeros((B, 1)), timestep], dim=1)
        num_clean = num_ref + num_cond_latents

        cache = transformer_options.get("longcat_clean_kv") if num_clean else None
        key = tuple(transformer_options["uuids"]) if cache is not None else None
        entry = cache.entries.get(key) if cache is not None else None

        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        t, h, w = (s // p for s, p in zip(x.shape[2:], self.patch_size))
        S = h * w
        # with the clean tokens' keys and values at hand, only the noisy frames run
        first = num_clean if entry is not None else 0
        img = self.x_embedder.proj(x[:, :, first:]).flatten(2).transpose(1, 2).contiguous()
        del x

        t_emb = self.t_embedder(timestep[:, first:].flatten(), dtype).view(B, t - first, -1)
        pe = self.rope(t, h, w, ref_index if num_ref else None, device)[:, first * S:]

        # ComfyUI's UMT5 zeroes the padding tokens; the release zeroes them after the projection (text_tokens_zero_pad)
        context = self.y_embedder.y_proj(context) * (context != 0).any(dim=-1, keepdim=True)

        audio = self.project_audio(audio_embeds)[:, num_cond_latents:].flatten(0, 1)

        groups = []
        if entry is not None:
            groups.append((0, (t - num_clean) * S, 0, t * S))
            clean_kv = entry[0]
        else:
            if num_ref:
                groups.append((0, S, 0, S))
            if num_cond_latents:
                groups.append((num_ref * S, num_clean * S, num_ref * S, num_clean * S))
            groups.append((num_clean * S, t * S, 0, t * S))
            clean_kv = [None if cache is None else [] for _ in self.blocks]

        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "single"
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for i, block in enumerate(self.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            img = block(img, context, audio, t_emb, pe, groups, num_clean - first, clean_kv[i], transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        img = self.final_layer(img, t_emb)
        if entry is not None:
            img = torch.cat([entry[1], img], dim=1)
        else:
            if cache is not None:
                cache.entries[key] = (clean_kv, img[:, num_ref * S:num_clean * S].clone())
            img = img[:, num_ref * S:]
        pt, ph, pw = self.patch_size
        # each token holds (pt, ph, pw, channels), channels innermost
        img = img.view(B, t - num_ref, h, w, pt, ph, pw, self.out_channels).permute(0, 7, 1, 4, 2, 5, 3, 6)
        img = img.reshape(B, self.out_channels, (t - num_ref) * pt, h * ph, w * pw)[:, :, :T, :H, :W]
        # the network predicts x0 - noise; ComfyUI's flow sampling wants noise - x0
        return -img
