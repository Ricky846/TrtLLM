# DeciCoder

This document explains how to build the DeciCoder using TensorRT-LLM and run on a single GPU.

## Overview

The TensorRT-LLM DeciCoder implementation can be found in [/workspace/tensorrt_llm_july-release-v1/examples/deci/decicoder/model.py](../deci/decicoder/model.py).The TensorRT-LLM DeciCoder example code is located in ['examples/deci'](./).The following files are included:

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the DeciCoder model,
 * [`run.py`](./run.py) to run the inference on an input text.
 * [`weight.py`](./weight.py) to load weight from the checkpoint file.

## Build and run DeciCoder on a single GPU

In this example, TensorRT-LLM builds TensorRT engine(s) from the [DeciCoder](https://huggingface.co/Deci/DeciCoder-1b) model.

Fisrt, you should download the files and checkpoints from the [huggingface](https://huggingface.co/Deci/DeciCoder-1b/tree/main) and put them in the folder ['examples/deci'](./). (put them anywhere that you want. Just make sure use right path for the file when you use the bulid and run command).Also you can use the following command to download the files:

```bash
git lfs install
git clone https://huggingface.co/Deci/DeciCoder-1b
```

Use the following command to build the TensorRT engine:

```bash
python build.py --model_dir ./DeciCoder-1b --dtype float16 --use_gemm_plugin float16 --use_identity_plugin float16 --output_dir ./trt_engines/fp16/1-gpu/
```

The following command can be used to run the DeciCoder model on a single GPU:

```bash
python run.py --input_text "def print_hello_world():" --max_output_len=50 --tokenizer_dir ./DeciCoder-1b --engine_dir=./trt_engines/fp16/1-gpu/
```