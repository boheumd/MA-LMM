"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel, BertSelfAttention, BertAttention, BertLayer, BertModel, BertEncoder
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from transformers import BertTokenizer
import math
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

class MBBertSelfAttention(BertSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            k = self.key(hidden_states)
            v = self.value(hidden_states)
            if hasattr(self, 'query_memory_bank'):
                B, T, N, C = self.query_memory_bank.shape
                query_memory_bank = self.query_memory_bank.view(B, -1, C) #[B, T*32, C]
                query_memory_bank_k = torch.cat([self.key(query_memory_bank), k], dim=1) #[B, (T+1)*32, C]
                query_memory_bank_v = torch.cat([self.value(query_memory_bank), v], dim=1) #[B, (T+1)*32, C]
                key_layer = self.transpose_for_scores(query_memory_bank_k)
                value_layer = self.transpose_for_scores(query_memory_bank_v)
            else:
                key_layer = self.transpose_for_scores(k)
                value_layer = self.transpose_for_scores(v)
        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None and not is_cross_attention:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            if hasattr(self, 'query_memory_bank'):
                attention_scores = attention_scores + torch.cat([attention_mask] * (self.query_memory_bank.size(1) + 1), dim=-1)
            else:
                attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        outputs = outputs + (past_key_value,)

        if not is_cross_attention:
            B, N, C = k.shape
            # if it is the first frame, initialize the query_memory_bank as the first frame's query embedding
            # if not, concatenate the query_memory_bank with the current frame embedding and update the compression_size
            if not hasattr(self, 'query_memory_bank'):
                self.query_memory_bank = hidden_states[:, None, :, :].detach() # [B, 1, 32, C]
                self.size_constant = torch.ones(B, 1, N).to(hidden_states.device) # [B, 1, N]
                self.compression_size = self.size_constant
            else:
                self.query_memory_bank = torch.cat([self.query_memory_bank, hidden_states[:, None, :, :].detach()], dim=1) # [B, t+1, 32, C]
                self.compression_size = torch.cat([self.compression_size, self.size_constant], dim=1) # [B, t+1, 32]

            # if it is the last frame, delete the query_memory_bank and compression_size
            # else if the current length of the query_memory_bank exceeds the threshold, compress the query_memory_bank
            if self.compression_size.sum(1).mean().round() == self.num_frames:
                del self.query_memory_bank
                del self.compression_size
            elif self.query_memory_bank.size(1) > self.memory_bank_length:
                self.query_memory_bank, self.compression_size = memory_bank_compress(self.query_memory_bank, self.compression_size)

        return outputs

class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, memory_bank_length=0, num_frames=0):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.memory_bank_length = memory_bank_length
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        Qformer.bert = apply_memory_bank(Qformer.bert, memory_bank_length, num_frames)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        # elif model_name == "eva2_clip_L":
        #     visual_encoder = create_eva2_vit_L(
        #         img_size, drop_path_rate, use_grad_checkpoint, precision
        #     )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        if self.vit_name == "eva_clip_g":
            vit_num_layers = self.visual_encoder.get_num_layer()
            lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

            parameter_group_names = {}
            parameter_group_vars = {}

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias"):
                    group_name = "no_decay"
                    this_weight_decay = 0.
                else:
                    group_name = "decay"
                    this_weight_decay = weight_decay
                if 'visual_encoder' in name:
                    layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                    group_name = "vit_layer_%d_%s" % (layer_id, group_name)
                else:
                    layer_id = None

                if group_name not in parameter_group_names:
                    if layer_id is not None:
                        scale = lr_scales[layer_id]
                    else:
                        scale = 1
                    parameter_group_names[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "lr_scale": scale
                    }
                    parameter_group_vars[group_name] = {
                        "weight_decay": this_weight_decay,
                        "params": [],
                        "lr_scale": scale
                    }
                parameter_group_vars[group_name]["params"].append(param)
                parameter_group_names[group_name]["params"].append(name)
            # import json
            # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
            optim_params = list(parameter_group_vars.values())
            return optim_params
        else:
            return super().get_optimizer_params(weight_decay,lr_scale)

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer



def memory_bank_compress(memory_bank: torch.Tensor, compression_size: torch.Tensor) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Compression_size is the number of frames that are compressed into each position.
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
        compression_size (torch.Tensor): The number of frames to compress into each position. Shape: (B, T, N)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
        compressed_size (torch.Tensor): The number of frames compressed into each position. Shape: (B, T-1, N)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    # Select the frame indices with the top-1 similarity 
    _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)

    # Calculate source and dst indices for compression
    src_indices = max_similarity_indices + 1
    dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)
    dst_indices[dst_indices > max_similarity_indices] += 1

    # Gather source and dst memory banks and sizes
    src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    src_size = compression_size.gather(dim=1, index=src_indices)
    dst_size = compression_size.gather(dim=1, index=dst_indices)

    # Multiply the memory banks by their corresponding sizes
    src_memory_bank *= src_size.unsqueeze(-1)
    dst_memory_bank *= dst_size.unsqueeze(-1)

    # Compress the memory bank by adding the source memory bank to the dst memory bank
    dst_memory_bank.scatter_add_(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src_memory_bank)
    dst_size.scatter_add_(dim=1, index=max_similarity_indices, src=src_size)

    # Normalize the dst memory bank by its size
    compressed_memory_bank = dst_memory_bank / dst_size.unsqueeze(-1)
    return compressed_memory_bank, dst_size

def apply_memory_bank(model, memory_bank_length, num_frames):
    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            if memory_bank_length > 0:
                module.__class__ = MBBertSelfAttention
                module.memory_bank_length = memory_bank_length
                module.num_frames = num_frames
    logging.info(str(model))
    return model


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
