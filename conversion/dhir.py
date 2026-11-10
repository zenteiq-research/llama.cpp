from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


def _set_vocab_dhir(model: TextModel) -> None:
    tokens, toktypes, tokpre = model.get_vocab_base()
    model.gguf_writer.add_tokenizer_model("gpt2")
    model.gguf_writer.add_tokenizer_pre(tokpre)
    model.gguf_writer.add_token_list(tokens)
    model.gguf_writer.add_token_types(toktypes)

    with open(model.dir_model / "tokenizer.json", encoding="utf-8") as f:
        added = json.load(f).get("added_tokens", [])
    ids = {t["content"]: t["id"] for t in added}

    # <eos> ends a document, <|end_of_turn|> ends a chat turn
    special_vocab = gguf.SpecialVocab(model.dir_model, load_merges=True)
    special_vocab.special_token_ids["eos"] = ids["<eos>"]
    special_vocab.special_token_ids["eot"] = ids["<|end_of_turn|>"]
    special_vocab.add_to_gguf(model.gguf_writer)

    if single_word := [t["id"] for t in added if t.get("single_word")]:
        model.gguf_writer.add_single_word_token_ids(single_word)


@ModelBase.register("DhirForCausalLM")
@ModelBase.example("zenteiq-ai/Dhir-2B")
class DhirModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DHIR

    def set_vocab(self):
        _set_vocab_dhir(self)


@ModelBase.register("DhirMoeForCausalLM")
@ModelBase.example("zenteiq-ai/Dhir-5B-A1.7B")
class DhirMoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DHIRMOE

    _experts: list[dict[str, Tensor]] | None = None

    def set_vocab(self):
        _set_vocab_dhir(self)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if ".mlp.experts." not in name:
            yield from super().modify_tensors(data_torch, name, bid)
            return

        # HF keeps one tensor per expert and projection; GGUF wants the experts of a layer stacked into one tensor
        assert bid is not None
        n_experts = self.hparams["num_experts"]
        if self._experts is None:
            self._experts = [{} for _ in range(self.block_count)]
        experts = self._experts[bid]
        experts[name] = data_torch
        if len(experts) < n_experts * 3:
            return

        for w_name in ("down_proj", "gate_proj", "up_proj"):
            names = [f"model.layers.{bid}.mlp.experts.{x}.{w_name}.weight" for x in range(n_experts)]
            data = torch.stack([experts.pop(n) for n in names], dim=0)
            yield from super().modify_tensors(data, f"model.layers.{bid}.mlp.experts.{w_name}.weight", bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None and (left := [k for d in self._experts for k in d]):
            raise ValueError(f"Unprocessed experts: {left}")
