#!/usr/bin/env python3
"""Export an official GPT-2 next-token model and its decode table.

The ONNX graph accepts fixed-length, right-padded GPT-2 input_ids and
attention_mask tensors. It selects the last non-padding position, applies the
language-model head, and emits one int64 next_token_id per batch element.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel


DECODE_MAGIC = b"GZGPT2D1"


class GPT2NextToken(nn.Module):
    def __init__(self, model: GPT2LMHeadModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]
        last_index = attention_mask.to(torch.int64).sum(dim=1) - 1
        batch_index = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[batch_index, last_index, :]
        return torch.argmax(last_logits, dim=-1)


def build_decode_table(tokenizer: AutoTokenizer, path: Path) -> dict[str, int]:
    offsets = [0]
    blob = bytearray()

    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        escaped = json.dumps(decoded, ensure_ascii=True)[1:-1].encode("ascii")
        blob.extend(escaped)
        offsets.append(len(blob))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<8sII", DECODE_MAGIC, len(tokenizer), len(blob)))
        stream.write(struct.pack(f"<{len(offsets)}I", *offsets))
        stream.write(blob)

    return {"vocab_size": len(tokenizer), "decode_bytes": len(blob)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--decode-table", type=Path, required=True)
    parser.add_argument("--prompt", default="The weather is")
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lm = GPT2LMHeadModel.from_pretrained(args.model).eval()
    wrapped = GPT2NextToken(lm).eval()

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.sequence_length,
    )
    input_ids = encoded["input_ids"].to(torch.int64)
    attention_mask = encoded["attention_mask"].to(torch.int64)

    with torch.inference_mode():
        torch_next = wrapped(input_ids, attention_mask)

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask),
        args.onnx,
        input_names=["input_ids", "attention_mask"],
        output_names=["next_token_id"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "next_token_id": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    graph = onnx.load(args.onnx)
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ort_next = session.run(
        ["next_token_id"],
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        },
    )[0]
    if int(ort_next[0]) != int(torch_next[0]):
        raise RuntimeError(
            f"next-token mismatch: PyTorch={int(torch_next[0])} ONNX={int(ort_next[0])}"
        )

    table = build_decode_table(tokenizer, args.decode_table)
    valid_count = int(attention_mask[0].sum())
    token_ids = input_ids[0, :valid_count].tolist()
    next_id = int(torch_next[0])
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "input_ids": token_ids,
        "wire_input": "ids:" + ",".join(str(token_id) for token_id in token_ids),
        "next_token_id": next_id,
        "next_token": tokenizer.decode(
            [next_id], clean_up_tokenization_spaces=False
        ),
        "onnx_check": "PASS",
        "pytorch_onnx_match": True,
        **table,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
