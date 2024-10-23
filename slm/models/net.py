"""Conditional Generative models defined for structure tokens used in ESM3.
"""
import os
import math
import torch 
from torch import nn
from torch import Tensor
from copy import deepcopy

import contextlib
from transformers import T5ForConditionalGeneration, T5Config, GPT2Model, GPT2Config
from esm.utils.constants import esm3 as C
from esm.models.esm3 import ESM3, ESMOutput, EncodeInputs, OutputHeads
from esm.layers.transformer_stack import TransformerStack
from esm.layers.regression_head import RegressionHead
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)
from esm.tokenization import get_model_tokenizers
from esm.utils.constants.models import (
    ESM3_OPEN_SMALL,
)
from esm.pretrained import (
    load_local_model, 
    ESM3_structure_encoder_v0, 
    ESM3_structure_decoder_v0, 
    ESM3_function_decoder_v0,
)

from slm.models.utils import cross_entropy


ESM3_D_MODEL = 1536
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 10000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def _shift_right(input_ids, pad_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = pad_token_id
    return shifted_input_ids

def _generate_square_subsequent_mask(
    sz: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    """Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


class CustomedGPT2(GPT2Model):
    """https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/gpt2"""
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        # load pretrained structure token embeddings (ESM3 VQ-VAE)
        _model_esm3 = ESM3.from_pretrained("esm3_sm_open_v1").to('cpu')
        _vq_decoder = _model_esm3.get_structure_decoder()
        self.structure_embed_tokens = deepcopy(_vq_decoder.embed)
        if config.freeze_dec_emb:
            for param in self.structure_embed_tokens.parameters():
                param.requires_grad = False
        del _vq_decoder, _model_esm3
        
        # input
        self.sequence_adapation_layer = nn.Linear(ESM3_D_MODEL, config.n_embd, bias=False)
        self.structure_adapation_layer = nn.Linear(1280, config.n_embd, bias=False)
        
        self.sequence_head = nn.Linear(config.n_embd, len(C.SEQUENCE_VOCAB), bias=False)     # leave one token for <sep>
        self.structure_head = nn.Linear(config.n_embd, C.VQVAE_CODEBOOK_SIZE + len(C.VQVAE_SPECIAL_TOKENS), bias=False)
        
        self.sep = config.sep_strategy
        if self.sep == 'sentence':
            pass
        elif self.sep == 'position':
            self.sep_token = nn.Parameter(torch.randn(config.n_embd))   # [D]
        else:
            raise ValueError(f"Unknown sequence strategy: {self.sep}")

        self.seq_loss_weight = config.seq_loss_weight if hasattr(config, 'seq_loss_weight') else 1.0    # historical for training

    def forward(self, **kwargs):
        training = kwargs.get('labels', None) is not None

        # always specified 
        B, L = kwargs['sequence_embeddings'].shape[:2]
        device = kwargs['sequence_embeddings'].device

        past_key_values = kwargs.get('past_key_values', None)
        # structure tokens may be partially provided during inference
        if past_key_values is None:
            L_structure = kwargs['structure_tokens'].shape[1]   # always >=1, inference starts with BOS
        else:
            L_structure = past_key_values[0][0].shape[-2] - L
        
        assert L_structure >= 1, f"L_structure must be >=1, but got {L_structure}"
        # concat sequence and structure embeddings
        structure_embeds = self.structure_embed_tokens(kwargs['structure_tokens'])

        if self.sep == 'sentence':  # (B, 2*L, D)
            # config sequence and structure ids
            token_type_ids = torch.cat([
                torch.zeros(B, L, dtype=torch.long, device=device),
                torch.ones(B, L_structure, dtype=torch.long, device=device),
            ], dim=1)
            input_embeds = torch.cat([
                self.sequence_adapation_layer(kwargs['sequence_embeddings']),   
                self.structure_adapation_layer(structure_embeds),
            ], dim=1)
            # https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/gpt2#transformers.GPT2Model.forward
            gpt_input_dict= {
                "inputs_embeds": input_embeds,
                "token_type_ids": token_type_ids,
                "return_dict": True,
                "use_cache": not training,
            }
            if 'mask' in kwargs:
                attn_mask = torch.cat([
                    kwargs['mask'],
                    kwargs['mask'][:, :L_structure],
                ], dim=1)
                gpt_input_dict["attention_mask"] = attn_mask 
        elif self.sep == 'position':    # (B, 2*L+1, D)
            # add sep token to sequence and structure
            input_embeds = torch.cat([
                self.sequence_adapation_layer(kwargs['sequence_embeddings']),   
                self.sep_token[None, None, :].expand(B, 1, -1),
                self.structure_adapation_layer(structure_embeds),
            ], dim=1)
            position_ids = torch.cat([
                torch.arange(L, dtype=torch.long, device=device)[None, :].expand(B, -1),
                torch.zeros(B, 1, dtype=torch.long, device=device),
                torch.arange(L_structure, dtype=torch.long, device=device)[None, :].expand(B, -1),
            ], dim=1)
        
            gpt_input_dict= {
                "inputs_embeds": input_embeds,
                "position_ids": position_ids,
                "return_dict": True,
                "use_cache": not training,
            }
            if 'mask' in kwargs:
                attn_mask = torch.cat([
                    kwargs['mask'],
                    torch.ones(B, 1, dtype=kwargs['mask'].dtype, device=device),
                    kwargs['mask'][:, :L_structure],
                ], dim=1)
                gpt_input_dict["attention_mask"] = attn_mask
        else:
            raise ValueError(f"Unknown sequence strategy: {self.sep}")
        
        
        # print("input_embeds", input_embeds.shape)
        if past_key_values:
            gpt_input_dict["inputs_embeds"] = gpt_input_dict["inputs_embeds"][:, -1:]  # [B, 1, D]
            
            if self.sep == 'sentence':
                gpt_input_dict["token_type_ids"] = gpt_input_dict["token_type_ids"][:, -1:]
            elif self.sep == 'position':
                gpt_input_dict["position_ids"] = gpt_input_dict["position_ids"][:, -1:]
            else:
                raise ValueError(f"Unknown sequence strategy: {self.sep}")

            outputs = super().forward(past_key_values=past_key_values, **gpt_input_dict)    
            h = outputs.last_hidden_state
            # print("h", h.shape)
            str_logits = self.structure_head(h[:, -1:])  # [B, V_struc]
            return_dict = {
                "structure_logits": str_logits, # different V
                "past_key_values": outputs.past_key_values,
            }
        else:
            outputs = super().forward(**gpt_input_dict)    
            h = outputs.last_hidden_state
            # print("h", h.shape)
            
            # split sequence and structure logits
            seq_logits = self.sequence_head(h[:, :L])   # [B, L, V_seq]
            str_logits = self.structure_head(h[:, L:])  # [B, L_structure, V_struc]
            
            return_dict = {
                "structure_logits": str_logits, # different V
                "sequence_logits": seq_logits,
                "past_key_values": outputs.past_key_values,
            }



        _labels = kwargs.get('labels', None)
        if _labels is not None: # full set of seq/structure tokens
            loss = 0
            if self.sep == 'sentence':
                loss_mask = kwargs['mask'][:, 1:]
            else:
                loss_mask = kwargs['mask']

            for name, logits, labels in [("sequence", seq_logits, _labels[:, :L]), ("structure", str_logits, _labels[:, L:])]:
                # print("shapes", logits.shape, labels.shape)

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                if self.sep == 'position' and name == 'structure':
                    shift_labels = labels
                    loss_mask = kwargs['mask']
                else:
                    shift_labels = labels[..., 1:].contiguous() # position: no shift, sep->label[0]
                    loss_mask = kwargs['mask'][:, 1:]

                _loss = cross_entropy(shift_logits, shift_labels, reduction='mean') # -100 is ignored for both vocab
                loss += _loss * self.seq_loss_weight if name == 'sequence' else _loss
                
                pred = torch.argmax(shift_logits, dim=-1) # [B, L, V] -> [B, L]
                acc = ((pred == shift_labels) * loss_mask).sum() / loss_mask.sum()
                return_dict[f"{name}_nll"] = _loss
                return_dict[f"{name}_acc"] = acc
            return_dict["loss"] = loss
        return return_dict



        
    
    


class CustomedT5(T5ForConditionalGeneration):
    """https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/t5/modeling_t5.py#L1432"""    
    def __init__(self, config: T5Config):
        super().__init__(config)
        # Customized lines
        if config.initialize_emb_from_vq:
            # load pretrained structure token embeddings (ESM3 VQ-VAE)
            _model_esm3 = ESM3.from_pretrained("esm3_sm_open_v1").to('cpu')
            _vq_decoder = _model_esm3.get_structure_decoder()
            self.decoder.embed_tokens = deepcopy(_vq_decoder.embed)
            assert config.d_model == self.decoder.embed_tokens.weight.size(1), f"{config.d_model} != {self.decoder.embed_tokens.weight.size(1)}"
            if config.freeze_dec_emb:
                for param in self.decoder.embed_tokens.parameters():
                    param.requires_grad = False

        self.adapation_layer = nn.Linear(ESM3_D_MODEL, config.d_model, bias=False)
        self.decoder_only = config.is_decoder
        if self.decoder_only: 
            print(">>> [CustomedT5] Decoder only mode")
        
        # Customized: also change inference time
        self.dec_add_input_emb = config.dec_add_input_emb
        
    def forward(self, **kwargs):
        kwargs['inputs_embeds'] = self.adapation_layer(kwargs['inputs_embeds'])
        
        if self.decoder_only:   # ignore encoder
            kwargs['encoder_outputs'] = [kwargs['inputs_embeds'], None, None]
            kwargs.pop('inputs_embeds')
        
        # customed decoder input
        if self.dec_add_input_emb:  
            # add condition embedding to decoder input (aligned to output position)
            # eg. dec 1 -> 2, dec input = 1 + c2
            if 'labels' in kwargs:
                decoder_input_ids = self._shift_right(kwargs['labels']) # training
                skip_cond = kwargs['inputs_embeds']
            else:
                decoder_input_ids = kwargs.pop('decoder_input_ids') # inference, [pad, 1, 2, 3, .., L-1] -> [1,2,3,...,L]
                skip_cond = kwargs['inputs_embeds'][:, :decoder_input_ids.size(1)]
            decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) + skip_cond
            kwargs['decoder_inputs_embeds'] = decoder_inputs_embeds
        return super().forward(**kwargs)


class StructureOutputHeads(nn.Module):
    def __init__(self, d_model: int, n_structure_heads: int = 4096, n_sequence_heads: int = 0):
        super().__init__()
        self.structure_head = RegressionHead(d_model, n_structure_heads)
        if n_sequence_heads:
            self.sequence_head = RegressionHead(d_model, n_sequence_heads)
        else:
            self.sequence_head = None

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
        structure_logits = self.structure_head(x)
        dummy_tensor = torch.zeros_like(structure_logits)
        sequence_logits = self.sequence_head(x) \
            if self.sequence_head is not None else dummy_tensor
        return ESMOutput(
            sequence_logits=sequence_logits,
            structure_logits=structure_logits,
            secondary_structure_logits=dummy_tensor,
            sasa_logits=dummy_tensor,
            function_logits=dummy_tensor,
            residue_logits=dummy_tensor,
            embeddings=embed,
        )

class CustomizedESM3(ESM3):
    def __init__(
        self, 
        d_model=1536,
        n_heads=24,
        v_heads=256,
        n_layers=48,
        pretrained=True,
        n_structure_heads=4096,
        n_sequence_heads=0, # >0 to enable sequence head
        *args,
        **kwargs,
    ):
        # config of ESM3_OPEN_SMALL
        super(ESM3, self).__init__()
        self.encoder = EncodeInputs(d_model)
        
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            v_heads,
            n_layers,
            mask_and_zero_frameless=True,
        )
        self.output_heads = OutputHeads(d_model)

        self.structure_encoder_fn = ESM3_structure_encoder_v0
        self.structure_decoder_fn = ESM3_structure_decoder_v0
        self.function_decoder_fn = ESM3_function_decoder_v0

        self._structure_encoder = None
        self._structure_decoder = None
        self._function_decoder = None

        self.tokenizers = get_model_tokenizers(ESM3_OPEN_SMALL)
        
        if pretrained:
            print("Load pretrained esm3 model...")
            model = load_local_model(ESM3_OPEN_SMALL) # if eval mode, will overwrite later
            self.load_state_dict(model.state_dict())
        
        if n_structure_heads != C.VQVAE_CODEBOOK_SIZE:
            # replace used head
            print(f">>> [CustomizedESM3] Replace output_heads() with n_structure_heads={n_structure_heads}, n_sequence_heads={n_sequence_heads}")
            self.output_heads = StructureOutputHeads(d_model, n_structure_heads, n_sequence_heads)
            
        self.d_model = d_model
        self.train()

    def forward(
        self, 
        structure_tokens: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sequence_tokens: torch.Tensor | None = None,    # for decoder
        *,
        encoder_embeddings: torch.Tensor | None = None,
        ss8_tokens: torch.Tensor | None = None,
        sasa_tokens: torch.Tensor | None = None,
        function_tokens: torch.Tensor | None = None,
        residue_annotation_tokens: torch.Tensor | None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        auxiliary_embeddings: torch.Tensor | None = None,
    ):
        """Forward pass and get loss. *Tailored for conformation generation task.*
        """

        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        chain_id = defaults(chain_id, 0)
        
        # non long dtype
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )
        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        if auxiliary_embeddings is not None:
            x = x + auxiliary_embeddings
        
        x, embedding = self.transformer(x, sequence_id, affine, affine_mask, chain_id)
        forward_output = self.output_heads(x, embedding) # ESMOutput
        
        if labels is not None:
            assert mask is not None, "mask must be provided when labels is not None"
            unreduced_loss = cross_entropy(forward_output.structure_logits, labels) # -100 is ignored
            loss = (unreduced_loss * mask).sum() / mask.sum()
            return_dict = {
                "decoder_embeddings": forward_output.embeddings,
                "structure_logits": forward_output.structure_logits,
                "sequence_logits": forward_output.sequence_logits,
                "loss": loss
            }
            return return_dict
        
        return forward_output   # inference
 

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=t.dtype)
            / half).to(device=t.device, dtype=t.dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
            [embedding,
            torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
