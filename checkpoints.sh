#!/bin/bash

git lfs clone https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers AugModel/models/checkpoints/llava-llama-3-8b
wget "https://drive.google.com/file/d/1e83wWQh9Tsficx0HM2rmc2KyYfBMpumf/view" -O clip_b16_grit1m_fultune_8xe.pth AugModel/models/checkpoints/