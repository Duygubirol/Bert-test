import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from datasets import load_from_disk, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    dataset_train = load_from_disk(args.training_dir)
    dataset_test = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(dataset_train)}")
    logger.info(f" loaded test_dataset length is: {len(dataset_test)}")

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Prepare model labels - useful in inference API
    labels = dataset_train.features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True if get_last_checkpoint(args.output_dir) is not None else False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        tokenizer=tokenizer,
    )

    # train model
    if get_last_checkpoint(args.output_dir) is not None:
        logger.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=dataset_test)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            print(f"{key} = {value}\n")

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
