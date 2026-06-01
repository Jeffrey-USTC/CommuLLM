---
library_name: peft
license: other
base_model: /home/liuzj/data/models/LLM-Research/Meta-Llama-3-8B
tags:
- base_model:adapter:/home/liuzj/data/models/LLM-Research/Meta-Llama-3-8B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: test_codec_real_output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# test_codec_real_output

This model is a fine-tuned version of [/home/liuzj/data/models/LLM-Research/Meta-Llama-3-8B](https://huggingface.co//home/liuzj/data/models/LLM-Research/Meta-Llama-3-8B) on the alpaca_en_demo dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- training_steps: 10

### Training results



### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.5.1+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2