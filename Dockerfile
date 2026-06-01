FROM 10.11.3.8:5000/user-images/new_joffee

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    "transformers>=4.49.0,<=4.56.2,!=4.52.0" \
    "datasets>=2.16.0,<=4.0.0" \
    "accelerate>=1.3.0,<=1.11.0" \
    "peft>=0.14.0,<=0.17.1" \
    "trl>=0.8.6,<=0.9.6" \
    "gradio>=4.38.0,<=5.45.0" \
    "matplotlib>=3.7.0" \
    "tyro<0.9.0" \
    "einops" \
    "numpy<2.0.0" \
    "pandas>=2.0.0" \
    "scipy" \
    "sentencepiece" \
    "tiktoken" \
    "modelscope>=1.14.0" \
    "hf-transfer" \
    "safetensors<=0.5.3" \
    "fire" \
    "omegaconf" \
    "packaging" \
    "protobuf" \
    "pyyaml" \
    "pydantic<=2.10.6" \
    "uvicorn" \
    "fastapi" \
    "sse-starlette" \
    "av" \
    "librosa" \
    "propcache!=0.4.0" \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    "nltk" \
    "jieba" \
    "rouge-chinese"