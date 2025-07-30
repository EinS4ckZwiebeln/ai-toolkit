from typing import Optional
from diffusers.models.attention_processor import Attention
import torch
import torch.nn.functional as F

# It's good practice to ensure the custom kernel is available.
try:
    from sageattention import sageattn
except ImportError:
    raise ImportError("Please install sageattention: pip install sage-attention")


class FixedFluxSageAttnProcessor2_0:
    """
    Corrected attention processor for Flux-like models using sageattention.

    This version fixes two critical issues:
    1.  It passes the required `sm_scale` to `sageattn` to prevent numerical instability.
    2.  It removes the misplaced output projections, conforming to the diffusers
        processor design pattern.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("This processor requires PyTorch 2.0 or newer.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        
        # Note: The `sageattention` kernel does not support attention masks or dropout.
        # This is generally acceptable for many training setups but is a limitation to be aware of.
        if attention_mask is not None:
            print("Warning: sageattention does not support attention_mask. It will be ignored.")

        batch_size, _, _ = hidden_states.shape
        inner_dim = attn.to_q(hidden_states).shape[-1]
        head_dim = inner_dim // attn.heads

        # 1. Self-attention projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 2. Cross-attention projections and concatenation (if needed)
        if encoder_hidden_states is not None:
            encoder_q = attn.add_q_proj(encoder_hidden_states)
            encoder_k = attn.add_k_proj(encoder_hidden_states)
            encoder_v = attn.add_v_proj(encoder_hidden_states)

            encoder_q = encoder_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_k = encoder_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_v = encoder_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_q = attn.norm_added_q(encoder_q)
            if attn.norm_added_k is not None:
                encoder_k = attn.norm_added_k(encoder_k)
            
            # Concatenate context and sample projections for joint attention
            query = torch.cat([encoder_q, query], dim=2)
            key = torch.cat([encoder_k, key], dim=2)
            value = torch.cat([encoder_v, value], dim=2)

        # 3. Apply Rotary Positional Embeddings
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Call sageattention with the critical fixes
        
        # *** FIX 1: Add the scaling factor ***
        # `F.scaled_dot_product_attention` applies this scale automatically.
        # `sageattn` requires it as an explicit `sm_scale` argument.
        # Without this, the dot products explode, causing NaNs, especially with quantization.
        sm_scale = attn.scale

        # The kernel expects an output tensor to write to.
        hidden_states = torch.empty_like(query)

        sageattn(
            q=query.contiguous(),  # Kernels often require contiguous tensors
            k=key.contiguous(),
            v=value.contiguous(),
            output=hidden_states,
            sm_scale=sm_scale,
            is_causal=False, # is_causal is always False in Flux's main transformer
        )

        # 5. Reshape and return
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # *** FIX 2: Remove misplaced projections ***
        # Output projections (`to_out`, `to_add_out`) are handled by the calling
        # `Attention` module, not inside the processor.
        
        if encoder_hidden_states is not None:
            # Split the output back into context and sample parts
            context_len = encoder_hidden_states.shape[1]
            encoder_hidden_states_out, hidden_states_out = (
                hidden_states[:, :context_len],
                hidden_states[:, context_len:],
            )
            return hidden_states_out, encoder_hidden_states_out
        else:
            return hidden_states