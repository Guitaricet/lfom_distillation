# FLOM distillation

T5 distillation in learning from other's mistakes way.

# TODO
[ ] Eval dropout05/distilt5_6l_8h_512d_2048ff on GLUE task
[ ]

# Installation

```bash
python -m pip install -r requirements.txt
```

# Run

> Note that downloading C4/en requires 1.5Tb storage and downloads over 350Gb. A smaller alternative is C4/realnewslike. 1 epochs should be enough 

Run `prepare_env.py` to create output dir and T5_dummy config.

## Step 1. Pre-train a dummy T5 model.

This model will be used as a baseline model that makes mistakes.

```bash
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR=pretrained_models/t5_2l_8h_512d_2048ff_vocab32128_fixed
export CACHE_DIR=/mnt/home/.cache/datasets
export WANDB_START_METHOD="thread"


python run_t5_mlm_flax.py \
    --output_dir=$MODEL_DIR \
    --model_type="t5" \
    --config_name="tiny_model_config" \
    --tokenizer_name="t5-large" \
    --dataset_name="c4" \
    --dataset_config_name="en" \
    --cache_dir $CACHE_DIR \
    --preprocessing_num_workers="128" \
    --max_seq_length="128" \
    --per_device_train_batch_size="1024" \
    --per_device_eval_batch_size="1024" \
    --adafactor \
    --learning_rate="0.01" \
    --weight_decay="0.001" \
    --warmup_steps="1024" \
    --overwrite_output_dir \
    --logging_steps="8" \
    --save_steps="1024" \
    --eval_steps="512" \
    --num_train_epochs "1" \
    --push_to_hub \
    --dataset_fraction="0.01"
```

Original T5 was pre-trained for `524,288` steps with batch size `128` and sequence length `512`. We cut this number by `64`, because our batch size is `32 * 8` times larger (8 devices) and sequence length is `4` times smaller which yields `8192` steps. Weirdly, it is slightly less than one epoch, so we decided to set it to one epoch exactly.


## Step 2. Distill T5

## LFOM Distillation

```bash
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR=pretrained_models/lfom_distilt5_6l_8h_512d_2048ff
export TEACHER_MODEL=t5-large
export WEAK_MODEL=pretrained_models/t5_2l_8h_512d_2048ff_vocab32128_fixed
export CACHE_DIR=/mnt/home/.cache/datasets

# REMEMBER TO REPLACE --config-name $WEAK_MODEL with a normal config
# REMEMBER TO ADD --weak_model_name_or_path

python run_lfom_distillation_flax.py \
    --output_dir=$MODEL_DIR \
    --model_type="t5" \
    --config_name="t5-small" \
    --tokenizer_name=$TEACHER_MODEL \
    --teacher_model_name_or_path=$TEACHER_MODEL \
    --weak_model_name_or_path=$WEAK_MODEL \
    --dataset_name="c4" \
    --dataset_config_name="en" \
    --cache_dir $CACHE_DIR \
    --preprocessing_num_workers="64" \
    --max_seq_length="256" \
    --temperature 2.0 \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --adafactor \
    --learning_rate="0.01" \
    --weight_decay="0.001" \
    --warmup_steps="1024" \
    --overwrite_output_dir \
    --logging_steps="8" \
    --save_steps="1024" \
    --eval_steps="512" \
    --num_train_epochs "1" \
    --push_to_hub \
    --dataset_fraction="0.1" # DEBUG option, make sure that validation set is still more that 1 element
```


To restart training:

```bash
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR=pretrained_models/lfom_distilt5_6l_8h_512d_2048ff_restarted
export TEACHER_MODEL=t5-large
export WEAK_MODEL=pretrained_models/t5_2l_8h_512d_2048ff_vocab32128_fixed
export CACHE_DIR=/mnt/home/.cache/datasets
export WANDB_START_METHOD="thread"

# REMEMBER TO REPLACE --config-name $WEAK_MODEL with a normal config
# REMEMBER TO ADD --weak_model_name_or_path

python run_lfom_distillation_flax.py \
    --output_dir=$MODEL_DIR \
    --model_type="t5" \
    --model_name_or_path="pretrained_models/lfom_distilt5_6l_8h_512d_2048ff" \
    --tokenizer_name=$TEACHER_MODEL \
    --teacher_model_name_or_path=$TEACHER_MODEL \
    --weak_model_name_or_path=$WEAK_MODEL \
    --dataset_name="c4" \
    --dataset_config_name="en" \
    --cache_dir $CACHE_DIR \
    --preprocessing_num_workers="64" \
    --max_seq_length="256" \
    --temperature 2.0 \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --adafactor \
    --learning_rate="0.01" \
    --weight_decay="0.001" \
    --warmup_steps="1024" \
    --overwrite_output_dir \
    --logging_steps="8" \
    --save_steps="1024" \
    --eval_steps="512" \
    --num_train_epochs="1" \
    --skip_train_steps="58368" \
    --push_to_hub \
    --dataset_fraction="0.1" # DEBUG option, make sure that validation set is still more that 1 element
```


## Distillation (baseline without LFOM)

```bash
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR=pretrained_models/distilt5_6l_8h_512d_2048ff
export TEACHER_MODEL=t5-large
export CACHE_DIR=/mnt/home/.cache/datasets
export WANDB_START_METHOD="thread"

python run_lfom_distillation_flax.py \
    --output_dir=$MODEL_DIR \
    --model_type="t5" \
    --config_name="t5-small" \
    --tokenizer_name=$TEACHER_MODEL \
    --teacher_model_name_or_path=$TEACHER_MODEL \
    --dataset_name="c4" \
    --dataset_config_name="en" \
    --cache_dir $CACHE_DIR \
    --preprocessing_num_workers="64" \
    --max_seq_length="256" \
    --temperature 2.0 \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --adafactor \
    --learning_rate="0.01" \
    --weight_decay="0.001" \
    --warmup_steps="1024" \
    --overwrite_output_dir \
    --logging_steps="8" \
    --save_steps="1024" \
    --eval_steps="512" \
    --num_train_epochs "1" \
    --push_to_hub \
    --dataset_fraction="0.1"
```

## Evaluate T5 model (upstream) without training

```bash
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR=pretrained_models/t5_base_eval
export CACHE_DIR=/mnt/home/.cache/datasets
export WANDB_START_METHOD="thread"


python run_t5_mlm_flax.py \
    --output_dir=$MODEL_DIR \
    --model_type="t5" \
    --config_name="t5-base" \
    --tokenizer_name="t5-large" \
    --dataset_name="c4" \
    --dataset_config_name="en" \
    --cache_dir $CACHE_DIR \
    --preprocessing_num_workers="128" \
    --max_seq_length="128" \
    --per_device_train_batch_size="1024" \
    --per_device_eval_batch_size="1024" \
    --dataset_fraction="0.1" \
    --do_eval_only

```

## Evaluate T5 model (downstream) on GLUE

```
export TOKENIZERS_PARALLELISM=false
export TASK_NAME=mrpc
export MODEL=dropout05/distilt5_6l_8h_512d_2048ff
export WANDB_START_METHOD="thread"


python run_flax_glue.py \
    --model_name_or_path $MODEL \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --eval_steps 100 \
    --output_dir finetuned/$TASK_NAME/

```
