import pickle
import argparse
from pathlib import Path

"""
Usage example:

python t5_classification_script_from_notebook.py \
    --model_name_or_path=dropout05/distilt5_realnewslike \
    --dataset_path="/home/vlialin/documents/biosbias/BIOS.pkl" \
    --output_dir=finetuned/distilt5_realnewslike

"""
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the BIOS.pkl file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Total batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Number of training steps to perform linear learning rate warmup for.")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading.")
    return parser.parse_args()


def cleanup_titles(examples):
    examples["title"] = examples["title"].replace("_", " ")
    return examples


if __name__ == "__main__":
    args = parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, from_flax=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    with open(args.dataset_path, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.train_test_split(0.2, seed=1)
    dataset = dataset.map(cleanup_titles)

    all_occupations = set(dataset["train"]["title"])
    encoded_titles = {}
    for title in set(dataset["train"]["title"]):
        with tokenizer.as_target_tokenizer():
            encoded_titles[title] = tokenizer(title, add_special_tokens=True, return_tensors="pt")["input_ids"]

    def preprocess(examples):
        # todo, do not remove gender field?
        input_encoding = tokenizer(examples["bio"])

        with tokenizer.as_target_tokenizer():
            lm_labels = tokenizer(examples["title"])["input_ids"]
            input_encoding["labels"] = lm_labels

        return input_encoding

    encoded_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        accuracy = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)

        # sanity check
        for l1, l2 in zip(decoded_labels, dataset["test"]["title"]):
            assert l1 == l2

        occupation2tpr_female = {}
        genders = dataset["test"]["gender"]
        for occupation in all_occupations:
            n_correct = sum(p == l for p, l, g in zip(decoded_preds, decoded_labels, genders) if l == occupation and g == "F")
            n_total = sum(1 for l, g in zip(decoded_labels, genders) if l == occupation and g == "F")
            occupation2tpr_female[occupation] = n_correct / n_total
        
        average_gap = 0
        occupation2gap = {}
        occupation2tpr_male = {}
        for occupation in all_occupations:
            n_correct = sum(p == l for p, l, g in zip(decoded_preds, decoded_labels, genders) if l == occupation and g == "M")
            n_total = sum(1 for l, g in zip(decoded_labels, genders) if l == occupation and g == "M")
            occupation2tpr_male[occupation] = n_correct / n_total

            gap = occupation2tpr_female[occupation] - occupation2tpr_male[occupation]
            occupation2gap[occupation] = gap
            average_gap += gap ** 2
        
        average_gap = np.sqrt(average_gap) / len(all_occupations)

        return {
            "accuracy": accuracy,
            "average_gap": average_gap,
            **{f"{o}_TPR/M": v for o, v in occupation2tpr_male.items()},
            **{f"{o}_TPR/F": v for o, v in occupation2tpr_female.items()},
        }

    args = Seq2SeqTrainingArguments(
        args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=10,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=args.eval_every,
        warmup_steps=args.warmup_steps,
        save_strategy="no",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
