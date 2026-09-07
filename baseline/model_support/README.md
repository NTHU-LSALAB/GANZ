# BERT and GPT model integration sources

This directory collects the model preparation and GAZSI integration sources for
BERT-base, GPT-2, and GPT-2-Large alongside the baseline materials.

| File | Purpose |
| --- | --- |
| `utils/build_bert_engine.py` | Export BERT-base and build an FP16 TensorRT engine with batch sizes 1–8 and sequence length 128. |
| `utils/export_gpt2_next_token.py` | Export GPT-2 or GPT-2-Large as a next-token ONNX model, produce its decode table, and generate a sample request using the model tokenizer. |
| `inference/gpu_pipeline.cu` and `.h` | GPU input preparation, parsing of pretokenized GPT-2 IDs, and embedding/next-token response construction. |
| `inference/tensorrt.cu` and `.h` | TensorRT model loading, model input/output handling, and GPT-2 decode-table integration. |

The GPT exporter selects the model with `--model` (`openai-community/gpt2` or
`openai-community/gpt2-large`). Its other required arguments are `--onnx` and
`--decode-table`. The generated report includes `wire_input` and the reference
next token. The runtime loads the corresponding decode table through
`GAZSI_GPT2_DECODE_TABLE`.

The four integration files are snapshots of the GAZSI implementation at commit
`9c9334a0ff0d9d0d55a2b8283b27ae89aee29c22`. Shared ring-buffer headers and support
code are in the repository's top-level `inference/` directory. These snapshots
are provided here for inspection; the default build still uses the top-level
sources. `../baseline_trt.cpp` remains the kernel TCP comparison implementation.

The experimental settings and published measurements are documented in
[REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) and the
[evaluation data supplement](../../data/evaluation/README.md).
