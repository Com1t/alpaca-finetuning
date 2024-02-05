# alpaca-finetuning

**Table Of Contents**

- [Description](#description)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
    - [Original Model weight](#original-model-weight)
    - [Finetuning ALPACA (`stanford_alpaca/train.py`)](#finetuning-alpaca-stanford_alpacatrainpy)
    - [Finetuning ALPACA LoRA (`alpaca-lora/finetune.py`)](#finetuning-alpaca-lora-alpaca-lorafinetunepy)
    - [Slurm environment](#slurm-environment)
- [Notes](#notes)

## Description

This repository is a simple guide about how to do finetuning using [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) and [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca.git).

A more thorogh instruction about how to do finetuning using these repository can be found in [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) and [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Prerequisites

1. Install dependencies (after you `cd` into those repo)

   ```bash
   pip install -r requirements.txt
   ```

2. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

NOTE: If you check the `requirements.txt`, it is the dependency needed by the repository. And since both are based on huggingface, `transformers` is the most crucial dependency.

## Running the sample

While there are many different usage to both repository, here I only interate over the process about host to do finetuning using each repo.
For other information, check the original repository.

### Original Model weight

Before you finetune the model, you need to download the model weight.
Although publicly accesible 'decapoda-research/llama-7b-hf' model weight provides competitive performance, if you want to use the original model weight from Meta, download the model weight from [meta-llama huggingface](https://huggingface.co/meta-llama) or [meta llama](https://ai.meta.com/llama/).

Note 1: Huggingface takes longer time to get authorization from meta, but it's easier to use.
Note 2: If you download the model weight from meta website, follow this [guide](https://huggingface.co/docs/transformers/main/model_doc/llama) to convert it into huggingface format.
Note 3: In case you're not familiar with huggingface. To use the downloaded model weight, you just need to set `--model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer>`. e.g., use model weight 'decapoda-research/llama-7b-hf' from huggingface


### Finetuning ALPACA (`stanford_alpaca/train.py`)

Example usage:

Below is a command that fine-tunes LLaMA-7B with our dataset on a machine with 4 A100 80G GPUs in FSDP `full_shard` mode.
Replace `<your_random_port>` with a port of your own, `<your_gpu_number>` with how many GPU do you have, `<your_path_to_hf_converted_llama_ckpt_and_tokenizer>` with the path to your converted checkpoint and tokenizer (following instructions in the PR), and `<your_output_dir>` with where you want to store your outputs.

```bash
torchrun --nproc_per_node=<your_gpu_number> --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

Naively, fine-tuning a 7B model requires about 7 x 4 x 4 = 112 GB of VRAM. Commands given above enable parameter sharding, so no redundant model copy is stored on any GPU.
If you'd like to further reduce the memory footprint, here are some options:

- Turn on CPU offload for FSDP with `--fsdp "full_shard auto_wrap offload"`. This saves VRAM at the cost of longer runtime.
- DeepSpeed stage-3 (with offload) can at times be more memory efficient than FSDP with offload. Here's an example to use DeepSpeed stage-3 with 4 GPUs with both parameter and optimizer offload:
```bash
(Remember to pip install deepspeed)
torchrun --nproc_per_node=<your_gpu_number> --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
```
- The DeepSpeed library also provides some [helpful functions](https://deepspeed.readthedocs.io/en/latest/memory.html) to estimate memory usage. 
- For more information about how to use deepspeed, check [Com1t/hf-slurm-training](https://github.com/Com1t/hf-slurm-training.git)

### Finetuning ALPACA LoRA (`alpaca-lora/finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.

Example usage:
This will finetune the base model `decapoda-research/llama-7b-hf` with the lora weights in `./lora-alpaca`.

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path './alpaca_data_cleaned_archive.json' \
    --output_dir './lora-alpaca'
```

We can also tweak our hyperparameters:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path './alpaca_data_cleaned_archive.json' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

To use other model weights:

Use model weight 'decapoda-research/llama-7b-hf' from huggingface
```bash
python finetune.py \
    --base_model decapoda-research/llama-7b-hf \
```

Use model weight '/work/llama-7b-chat' from local directory
```bash
python finetune.py \
    --base_model /work/llama-7b-chat \
```

To use more GPU:

This implementation supports multi-GPU, and doesn't require any modification to the running command.

### Slurm environment

Simply concatenate the above commands (those include and after `xxx.py`) to the end of this SBATCH script.
For more information about how to use slurm, check [Com1t/hf-slurm-training](https://github.com/Com1t/hf-slurm-training.git)

```
#!/bin/bash
#SBATCH --job-name=gpt2_multi    ## job name
#SBATCH --nodes=2                ## request 2 nodes
#SBATCH --ntasks-per-node=1      ## run 1 srun task per node
#SBATCH --cpus-per-task=32       ## allocate 32 CPUs per srun task
#SBATCH --gres=gpu:8             ## request 8 GPUs per node
#SBATCH --time=00:10:00          ## run for a maximum of 10 minutes
#SBATCH --account="XXX"          ## PROJECT_ID, please fill in your project ID (e.g., XXX)
#SBATCH --partition=gp1d         ## gtest is for testing; you can change it to gp1d (1-day run), gp2d (2-day run), gp4d (4-day run), etc.
#SBATCH -o %j.out                # Path to the standard output file
#SBATCH -e %j.err                # Path to the standard error output file

module purge
module load pkg/Anaconda3 cuda/11.7 compiler/gcc/11.2.0

conda activate alpaca

nvidia-smi

export GPUS_PER_NODE=8

# weights and bias API key
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
export MASTER_PORT=$(shuf -i 30000-60000 -n1)
export RDZV_ID=$(shuf -i 10-60000 -n1)

echo "NODELIST="${SLURM_JOB_NODELIST}
echo "MASTER_ADDR="${MASTER_ADDR}
echo "MASTER_PORT="${MASTER_PORT}

export OUTPUT_DIR=${HOME}/GPT_DDP_weights

srun bash -c \
  'torchrun --nnodes ${SLURM_NNODES} \
  --node_rank ${SLURM_PROCID} \
  --nproc_per_node ${GPUS_PER_NODE} \
  --rdzv_id ${RDZV_ID} \
  --rdzv_backend c10d \
  --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
```

## Notes

- For a more complete usage instruction, check [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) and [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca.git).
