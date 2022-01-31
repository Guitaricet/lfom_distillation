# FLOM distillation

T5 distillation in learning from other's mistakes way.

# Installation

```bash
python -m pip install -r requirements.txt
```

# Run

> Note that downloading C4/en requires 1.5Tb storage and downloads over 350Gb. A smaller alternative is C4/realnewslike. 1 epochs should be enough 

Run `prepare_env.py` to create output dir and T5_dummy config.

### Step 1. Pre-train a dummy T5 model.

This model will be used as a baseline model that makes mistakes.

```bash
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR=pretrained_models/t5_2l_8h_512d_2048ff
export CACHE_DIR=/mnt/home/.cache/datasets


python run_t5_mlm_flax.py \
	--output_dir=$MODEL_DIR \
	--model_type="t5" \
	--config_name=$MODEL_DIR \
	--tokenizer_name="t5-large" \
	--dataset_name="c4" \
	--dataset_config_name="en" \
    --cache_dir $CACHE_DIR \
	--preprocessing_num_workers="32" \
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
    --num_train_steps "8192" \
	--dataset_fraction="0.01" # DEBUG option, make sure that validation set is still more that 1 element
```

Original T5 was pre-trained for `524,288` steps with batch size `128` and sequence length `512`. We cut this number by `64`, because our batch size is `32 * 8` times larger (8 devices) and sequence length is `4` times smaller which yields `8192` steps.


### Step 2. Distill T5

# LFOM Distillation

```bash
export TOKENIZERS_PARALLELISM=false
export MODEL_DIR=pretrained_models/t5_2l_8h_512d_2048ff_lfom_distil_debug
export TEACHER_MODEL=t5-small
export WEAK_MODEL=t5_2l_8h_512d_2048ff

# REMEMBER TO REPLACE --config-name $WEAK_MODEL with a normal config
# REMEMBER TO ADD --weak_model_name_or_path

python run_lfom_distillation_flax.py \
	--output_dir=$MODEL_DIR \
	--model_type="t5" \
	--config_name=$WEAK_MODEL \
	--tokenizer_name=$TEACHER_MODEL \
	--teacher_model_name_or_path=$TEACHER_MODEL \
    --weak_model_name_or_path=$WEAK_MODEL \
	--dataset_name="c4" \
	--dataset_config_name="realnewslike" \
	--preprocessing_num_workers="8" \
	--max_seq_length="256" \
	--temperature 2.0 \
	--per_device_train_batch_size="128" \
	--per_device_eval_batch_size="128" \
	--adafactor \
	--learning_rate="0.01" \
	--weight_decay="0.001" \
	--warmup_steps="1024" \
	--overwrite_output_dir \
	--logging_steps="8" \
	--save_steps="1024" \
	--eval_steps="512" \
  --num_train_steps "8192" \
	--dataset_fraction="0.1" # DEBUG option, make sure that validation set is still more that 1 element
```
