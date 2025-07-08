

-----

# Fine-Tuning a Qwen LLM on LANTA with MS-Swift

This guide provides a comprehensive walkthrough for fine-tuning a Qwen language model using the MS-Swift library on the LANTA HPC cluster. We will cover environment setup, data preparation, and submitting the fine-tuning job.

## Table of Contents

1.  [Prerequisites](#1-prerequisites)
2.  [Step 1: Environment Setup on LANTA](#step-2-downloading-the-pre-trained-model)
3.  [Step 2: Downloading the Pre-trained Model](#step-3-data-preparation)
4.  [Step 3: Data Preparation](#step-3-data-preparation)
5.  [Step 4: The Fine-Tuning Script](#step-4-the-fine-tuning-script)
6.  [Step 5: Running Inference (After Training)](#step-5-running-inference-after-training)
-----

### 1\. Prerequisites

You must have an account on the LANTA cluster with access to a project space and GPU nodes.

### Step 1: Environment Setup on LANTA

First, log in to a LANTA transfer or compute node. We will create a dedicated Mamba environment to manage our dependencies.

1.  **Load the Mamba Module:**

    ```bash
    ml mamba
    ```

2.  **Create a Mamba Environment:**
    We will clone a pre-existing PyTorch environment to ensure compatibility with the system's CUDA drivers.

    ```bash
    mamba create --name ms-swift --clone pytorch-2.2.2
    ```

3.  **Activate the Environment:**
    Always activate your environment before installing packages or running scripts.

    ```bash
    mamba activate ms-swift
    ```

4.  **Install `ms-swift` and Dependencies:**
    Install `ms-swift` with the recommended dependencies. The `[llm]` extra includes all necessary packages for LLM fine-tuning.

    ```bash
    # (Optional) Set a pip mirror for faster downloads
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

    # Install ms-swift with LLM capabilities
    pip install 'ms-swift[llm]' -U

    # Install other crucial packages
    pip install transformers deepspeed flash-attn --no-build-isolation
    ```

      * **`transformers`**: The core library from Hugging Face.
      * **`deepspeed`**: Essential for efficient multi-GPU training.
      * **`flash-attn`**: An optimized attention mechanism to accelerate training and save memory.

### Step 2: Downloading the Pre-trained Model

We will use the `modelscope` library to download our base model. To avoid cluttering your home directory, we'll configure the cache to point to your project space.

1.  **Set ModelScope Cache Environment Variables:**
    Execute these commands in your shell, and consider adding them to your `~/.bashrc` file for convenience. **Remember to replace the path** with your actual project directory.

    ```bash
    # This tells modelscope where to save downloaded models
    export MS_CACHE_HOME="/project/lt200344-zhthmt/Y/.cache/modelscope"
    export MODELSCOPE_CACHE="/project/lt200344-zhthmt/Y/.cache/modelscope"
    ```

2.  **Download the Model:**
    Use the `modelscope` CLI to download the model. This command will download the `Qwen/Qwen3-0.6B` model to the directory specified by `MS_CACHE_HOME`.

    ```bash
    # Ensure you are in your ms-swift mamba environment
    modelscope download --model Qwen/Qwen3-0.6B
    ```

    For the final script, we will be using a larger `Qwen3-8B` model. You can download it using the same method:

    ```bash
    modelscope download --model Qwen/Qwen3-8B --local_dir /project/lt200344-zhthmt/Y/.cache/modelscope/models/Qwen/Qwen3-8B
    ```

### Step 3: Data Preparation

LLM fine-tuning requires data in a specific conversational format. The script below converts a simple JSONL file with "source" and "translation" fields into the required format for MS-Swift.

**Input Format (`train.jsonl`):**

```json
{"context": "...", "source": "哦，那还是扁桃体发炎了，你经常嗓子疼吗？", "translation": "ต่อมทอนซิลอักเสบแล้ว เจ็บคอบ่อยไหม"}
```

**Target Format (`ready_to_train.jsonl`):**

```json
{"system": "你是一个优秀的中泰医疗翻译师", "conversation": [{"human": "哦，那还是扁桃体发炎了，你经常嗓子疼吗？", "assistant": "ต่อมทอนซิลอักเสบแล้ว เจ็บคอบ่อยไหม"}]}
```

**Python Conversion Script:**
Save this code as `prepare_data.py`.

```python
import json
from tqdm import tqdm

# Define your input and output paths
input_path = "/project/lt200344-zhthmt/Y/MS-SWIFT/data/train.jsonl"
output_path = "/project/lt200344-zhthmt/Y/MS-SWIFT/data/ready_to_train.jsonl"

def format_data():
    """
    Transforms the raw JSONL data into the MS-Swift conversational format.
    """
    try:
        # Count total lines for a nice progress bar
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        # Open files for reading and writing
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:

            # Use tqdm for progress tracking
            for line in tqdm(infile, total=total_lines, desc="Processing Data"):
                try:
                    data = json.loads(line)
                    source = data.get("source", "").strip()
                    translation = data.get("translation", "").strip()

                    # Define the system prompt and conversation structure
                    new_line = {
                        "system": "你是一个优秀的中泰医疗翻译师",
                        "conversation": [
                            {"human": source, "assistant": translation}
                        ]
                    }
                    # Write the transformed data to the output file
                    outfile.write(json.dumps(new_line, ensure_ascii=False) + "\n")

                except json.JSONDecodeError:
                    print(f"⚠️ Warning: Skipped one line due to a JSON decoding error.")
        print(f"✅ Data preparation complete. Output saved to {output_path}")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_path}")

if __name__ == "__main__":
    format_data()
```

Run the script from your terminal:

```bash
python prepare_data.py
```

### Step 4: The Fine-Tuning Script

We will use a SLURM batch script to submit our training job to the LANTA GPU partition. This script handles resource allocation, environment activation, and executing the `ms-swift` training command.

**Review of Key Parameters:**

  * `--train_type lora`: We use Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. This drastically reduces memory usage compared to full fine-tuning.
  * `--per_device_train_batch_size 8`: Batch size per GPU.
  * `--gradient_accumulation_steps 4`: Accumulates gradients over 4 steps, effectively simulating a larger global batch size of `8 * 4 GPUs * 4 steps = 128`.
  * `--target_modules all-linear`: Applies LoRA to all linear layers in the model, a common and effective strategy.
  * `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`: An important setting for newer PyTorch/Transformers versions to save memory.
  * `--dataloader_num_workers`: **Set to a reasonable number.** A value like 16 or 32 is much safer than 256, which can cause system overhead. It should be based on the allocated CPUs (`-c 32`).

**SLURM Job Script (`run_sft.sh`):**

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
#SBATCH -A lt200344
#SBATCH -J ms-swift-qwen3-8b-sft

# --- Environment Setup ---
ml Mamba
source activate ms-swift

# --- Logging ---
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qwen3-8b_lora_sft_${TIMESTAMP}.log"

# --- Distributed Training Setup ---
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting MS-Swift SFT Job..."
echo "Log file will be saved to: $LOG_FILE"

# --- Supervised Fine-Tuning Command ---
swift sft \
    --model '/project/lt200344-zhthmt/Y/.cache/modelscope/models/Qwen/Qwen3-8B' \
    --train_type lora \
    --dataset '/project/lt200344-zhthmt/Y/MS-SWIFT/data/ready_to_train.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --split_dataset_ratio 0 \
    --lora_rank 16 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --weight_decay 0.01 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir ../swift_output/MDwen3-8B-Lora-SFT \
    --dataloader_num_workers 16 \
    --model_author sakchaisaehoei \
    --model_name MDwen3-8B-Lora-SFT \
    > "$LOG_FILE" 2>&1

echo "Job finished. Check logs for details."
```

**To submit the job:**

```bash
sbatch run_sft.sh
```

**To monitor the job:**

```bash
# Check the queue
squeue -u $USER

# View the log file in real-time
tail -f logs/qwen3-8b_lora_sft_*.log
```

### Step 5: Running Inference (After Training)

Once your fine-tuning is complete, the LoRA adapter weights will be saved in the `--output_dir` you specified. You can then use the `swift infer` command to test your model.

```bash
swift infer \
    --ckpt_dir '../swift_output/MDwen3-8B-Lora-SFT/checkpoint-xxxx' \ # Replace with your best checkpoint
    --load_dataset_config true # Use the tokenizer/template config from the checkpoint
```
