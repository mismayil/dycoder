# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from dotenv import load_dotenv

from utils import BatchComputeRangeIterator, ComputeRange

load_dotenv()

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Dycoder(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Dycoder, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        batch_compute_ranges = []

        for b in range(input_ids.shape[0]):
            compute_ranges = []
            
            start_latent_positions = (
                input_ids[b] == self.start_latent_id
            ).nonzero().reshape(-1).tolist() 
            end_latent_positions = (
                input_ids[b] == self.end_latent_id
            ).nonzero().reshape(-1).tolist()
            
            all_positions = sorted(start_latent_positions + end_latent_positions)
            
            if all_positions:
                if all_positions[0] > 0:
                    compute_ranges.append(ComputeRange(0, all_positions[0], "lang"))

                for i in range(len(all_positions)-1):
                    if i % 2 == 0:
                        compute_ranges.append(ComputeRange(all_positions[i], all_positions[i+1], "latent"))
                    else:
                        compute_ranges.append(ComputeRange(all_positions[i], all_positions[i+1], "lang"))
                
                if all_positions[-1] < input_ids.shape[1]:
                    compute_ranges.append(ComputeRange(all_positions[-1], input_ids.shape[1], "lang"))
                
                batch_compute_ranges.append(compute_ranges)
            else:
                batch_compute_ranges.append([ComputeRange(0, input_ids.shape[1], "lang")])

        batch_cr_iterator = BatchComputeRangeIterator(batch_compute_ranges)
        inputs_embeds = self.embedding(input_ids)
        batch_inputs_embeds = [inputs_embeds[b] for b in range(input_ids.shape[0])]
        batch_logits = [None] * input_ids.shape[0]

        for (lang_batch_indices, lang_compute_range), (latent_batch_indices, latent_compute_range) in batch_cr_iterator:
            
            if lang_batch_indices is not None:
                lang_inputs_embeds = torch.stack([batch_inputs_embeds[b][: lang_compute_range.end, :] for b in lang_batch_indices])
                lang_attention_mask = torch.stack([attention_mask[b, : lang_compute_range.end] for b in lang_batch_indices])
                lang_position_ids = torch.stack([position_ids[b, : lang_compute_range.end] for b in lang_batch_indices])
                outputs = self.base_causallm(
                    inputs_embeds=lang_inputs_embeds,
                    attention_mask=lang_attention_mask,
                    position_ids=lang_position_ids,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states

                for idx, b in enumerate(lang_batch_indices):
                    batch_logits[b] = outputs.logits[idx]

            if latent_batch_indices is not None:
                for latent_id in range(latent_compute_range.start+1, latent_compute_range.end):
                    latent_inputs_embeds = torch.stack([batch_inputs_embeds[b][: latent_id, :] for b in latent_batch_indices])
                    latent_attention_mask = torch.stack([attention_mask[b, : latent_id] for b in latent_batch_indices])
                    latent_position_ids = torch.stack([position_ids[b, : latent_id] for b in latent_batch_indices])
                    outputs = self.base_causallm(
                        inputs_embeds=latent_inputs_embeds,
                        attention_mask=latent_attention_mask,
                        position_ids=latent_position_ids,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states
                    for idx, b in enumerate(latent_batch_indices):
                        batch_inputs_embeds[b][latent_id] = hidden_states[idx][latent_id - 1, :]
                        batch_logits[b] = outputs.logits[idx]

                self.gen_forward_cnt += (latent_compute_range.end - latent_compute_range.start)

        logits = torch.stack(batch_logits)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds
        latent_mode = False

        # get other tokens
        for _ in range(max_new_tokens):
            outputs = self.base_causallm(inputs_embeds=inputs_embeds, output_hidden_states=True)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            
            if next_token == self.eos_token_id:
                break
            
            if next_token == self.end_latent_id:
                latent_mode = False
                new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            elif latent_mode:
                # replace with the preceding last hidden states
                new_token_embed = outputs.hidden_states[-1][-1][-1].view(1, 1, -1)
            elif next_token == self.start_latent_id:
                latent_mode = True
                new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            else:
                new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            
            tokens.append(next_token)
            inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)
