#!/usr/bin/env python3
"""Build the BERT-base TensorRT engine used by the fresh GAZSI validation."""

from pathlib import Path

import tensorrt as trt
import torch
from transformers import AutoModel


OUTPUT_DIR = Path(__file__).resolve().parent
ONNX_PATH = OUTPUT_DIR / "bert_base_batch1_8_seq128.onnx"
ENGINE_PATH = OUTPUT_DIR / "bert_base_batch1_8_seq128_fp16.engine"
MODEL_NAME = "bert-base-uncased"
SEQUENCE_LENGTH = 128
MAX_BATCH_SIZE = 8


class BertOutputs(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        result = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return result.last_hidden_state, result.pooler_output


def export_onnx() -> None:
    if ONNX_PATH.exists():
        print(f"Reusing {ONNX_PATH}")
        return

    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    wrapper = BertOutputs(model).eval()
    input_ids = torch.zeros((1, SEQUENCE_LENGTH), dtype=torch.int64)
    attention_mask = torch.ones((1, SEQUENCE_LENGTH), dtype=torch.int64)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            ONNX_PATH,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
                "pooler_output": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"Exported {ONNX_PATH}")


def build_engine() -> None:
    if ENGINE_PATH.exists():
        print(f"Reusing {ENGINE_PATH}")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(ONNX_PATH)):
        messages = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(messages))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for name in ("input_ids", "attention_mask"):
        profile.set_shape(
            name,
            min=(1, SEQUENCE_LENGTH),
            opt=(4, SEQUENCE_LENGTH),
            max=(MAX_BATCH_SIZE, SEQUENCE_LENGTH),
        )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    ENGINE_PATH.write_bytes(serialized)
    print(f"Built {ENGINE_PATH} ({ENGINE_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    export_onnx()
    build_engine()
