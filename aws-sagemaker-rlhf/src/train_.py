import mlflow
import datetime
import argparse
import sys
import logging
import os
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_from_disk
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftConfig, PeftModel


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

# hyperparameters sent by the client are passed as command-line arguments to the script.
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--micro_batch_size", type=int, default=32)
parser.add_argument("--warmup_steps", type=int, default=6)
parser.add_argument("--save_steps", type=int, default=35)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--seq_len", type=int, default=256)
parser.add_argument("--lora_r", type=int, default=4)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)

# Data, model, and output directories
parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
parser.add_argument("--output_model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
parser.add_argument("--model_dir", type=str, default=os.environ["SM_CHANNEL_MODEL"])

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
    MICRO_BATCH_SIZE=args.micro_batch_size
    BATCH_SIZE=args.batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS=args.epochs
    LEARNING_RATE=args.learning_rate
    WARMUP_STEPS=args.warmup_steps
    SAVE_STEPS=args.save_steps
    MODEL_DIR=args.model_dir
    TRAIN_DIR = args.train_dir
    VALIDATION_DIR = args.test_dir
    OUTPUT_DATA_DIR = args.output_data_dir
    OUTPUT_MODEL_DIR = args.output_model_dir

    SEQ_LEN = args.seq_len
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout

    # Set run name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%M%H%S")
    RUN_NAME = f"gpt2-mlflow-{timestamp}"
    mlflow.set_tag("mlflow.runName", RUN_NAME)

    # Log hyperparameters
    mlflow.log_params({
        'batch_size': BATCH_SIZE,
        'micro_batch_size': MICRO_BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'sequence_length': SEQ_LEN,
        'lora_rank': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT
    })

    # Tokenizer + Model + Data Collator
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load datasets
    train_dataset = load_from_disk(TRAIN_DIR)
    validation_dataset = load_from_disk(VALIDATION_DIR)
    train_mlflow = mlflow.data.from_huggingface(train_dataset)
    validation_mlflow = mlflow.data.from_huggingface(validation_dataset)
    mlflow.log_input(train_mlflow, context="training")
    mlflow.log_input(validation_mlflow, context="validation")

    print("Logged dataset")
    logger.info("Logged dataset")

    # Tokenizer + Model + Data Collator
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
    tokenizer.save_pretrained("/opt/ml/model/")
    logger.info(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb_config, device_map="auto")
    logger.info(model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. must be different from the eos token

    print("Model and tokenizer loaded")
    logger.info("Model and tokenizer loaded")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        fp16=True,
        logging_steps=100,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        #weight_decay=0.01,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        #save_steps=SAVE_STEPS,
        save_strategy="epoch",
        #save_total_limit=5,
        logging_dir=OUTPUT_DATA_DIR,
        #max_steps=5
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
    # Saves the adapter_config.json and adapter_model.bin to /opt/ml/model/
    trainer.model.save_pretrained(OUTPUT_MODEL_DIR)

    # clear memory
    del model
    del trainer
    # load PEFT adapter config from /opt/ml/model/ and model from /opt/ml/input/data/model/ in fp16.
    peft_config = PeftConfig.from_pretrained(OUTPUT_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        return_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # load PEFT adapter_model.bin from /opt/ml/model/
    model = PeftModel.from_pretrained(model, OUTPUT_MODEL_DIR)
    model.eval()
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_MODEL_DIR)

    # copy inference script
    os.makedirs(f"{OUTPUT_MODEL_DIR}/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        f"{OUTPUT_MODEL_DIR}/code/inference.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        f"{OUTPUT_MODEL_DIR}/code/requirements.txt",
    )

    # In the case of Dolly.
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "instruction_pipeline.py"),
        f"{OUTPUT_MODEL_DIR}/code/instruction_pipeline.py"
    )

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
    mlflow.log_artifacts(OUTPUT_MODEL_DIR)

    print("Logged artifacts")
    logger.info("Logged artifacts")