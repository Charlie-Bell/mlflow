import mlflow
import datetime
import argparse
import sys
import logging
import os
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tracking_uri", type=str)
parser.add_argument("--experiment_name", type=str, default="Default")
parser.add_argument("--s3_dir", type=str)
args, _ = parser.parse_known_args()

tracking_uri = args.tracking_uri
experiment_name = args.experiment_name
s3_dir = args.s3_dir

# Set remote mlflow server
mlflow.set_tracking_uri(tracking_uri)
print(tracking_uri)
logger.info(tracking_uri)
print(s3_dir)
logger.info(s3_dir)

# Set experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
if not experiment:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Start run
with mlflow.start_run() as run:
    # Configure hyperparameters
    EOS_TOKEN='<|endoftext|>'
    BATCH_SIZE=4
    EPOCHS=1
    LEARNING_RATE=2e-5
    SAVE_STEPS=625
    MODEL_PATH='distilgpt2'
    TRAINING_PATH = os.environ["SM_CHANNEL_TRAIN"]
    VALIDATION_PATH = os.environ["SM_CHANNEL_TEST"]

    # Set run name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%M%H%S")
    RUN_NAME = f"gpt2-mlflow-{timestamp}"
    mlflow.set_tag("mlflow.runName", RUN_NAME)

    # Log hyperparameters
    mlflow.log_params({
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
    })

    # Load datasets
    train_dataset = load_from_disk(TRAINING_PATH)
    validation_dataset = load_from_disk(VALIDATION_PATH)
    train_mlflow = mlflow.data.from_huggingface(train_dataset)
    validation_mlflow = mlflow.data.from_huggingface(validation_dataset)
    mlflow.log_input(train_mlflow, context="training")
    mlflow.log_input(validation_mlflow, context="validation")

    print("Logged dataset")
    logger.info("Logged dataset")

    # Tokenizer + Model + Data Collator
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2', return_tensors='pt', eos_token=EOS_TOKEN, pad_token=EOS_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Model and tokenizer loaded")
    logger.info("Model and tokenizer loaded")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/opt/ml/checkpoints",
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_steps=SAVE_STEPS,
        save_total_limit=5,
        logging_dir="/opt/ml/logs",
        max_steps=4
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    print("Training start")
    logger.info("Training start")

    # Train
    trainer.train()
    trainer.save_model("/opt/ml/model")

    print("Saved model")
    logger.info("Saved model")
    
    # Log metrics
    history = trainer.state.log_history
    mlflow.log_metric('epoch', history[-2]['eval_loss'])
    result = history[-1]
    for metric in result.keys():
        mlflow.log_metric(metric, result[metric])

    print("Logged metrics")
    logger.info("Logged metrics")

    # Log model + environment
    mlflow.log_artifacts("/opt/ml/model")

    print("Logged artifacts")
    logger.info("Logged artifacts")