#! /bin/bash
python /srcs/sources/llm/TensorRT-LLM/examples/run.py \
    --tokenizer_dir /srcs/.cache/modelscope/hub/models/Qwen/Qwen1___5-1___8B-Chat/ \
    --engine_dir ./xiaoan_engine/ \
    --input_text "my name is" \
    --max_output_len 1