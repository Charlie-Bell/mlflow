import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3
import dotenv
import os

dotenv.load_dotenv()

# Create sessions and client
boto_session = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=boto_session)
client = boto_session.client('sagemaker')
s3 = boto3.client('s3')

# Define variables
tracking_uri = os.environ.get("TRACKING_URI")
role = os.environ.get("ROLE")
s3_dir = os.environ.get("S3_DIR")
subnet = os.environ.get("SUBNET")
security_group_id = os.environ.get("SECURITY_GROUP_ID")
experiment_name = "experiment-2"

# Create estimator
DATASET_DIR = "s3://untapped-transformer-models/amazon_qa_dataset/closed_dataset/"
#MODEL_DIR = "s3://untapped-transformer-models/models/base/dolly-v2-12b/"
TRAIN_PATH = DATASET_DIR + "train"
VALIDATION_PATH = DATASET_DIR + "validation"
OUTPUT_DIR = "s3://untapped-transformer-models/models/trained/gpt-2-mlflow/"
ROLE = role

# Set arguments
hyperparameters={
    'SAGEMAKER_SUBMIT_DIRECTORY': OUTPUT_DIR,
    #'input_model_dir': MODEL_DIR,
    "tracking_uri": tracking_uri,
    "experiment_name": experiment_name,
    "s3_dir": s3_dir,
}

huggingface_estimator = HuggingFace(
    entry_point="train.py",                 # fine-tuning script to use in training job
    source_dir="./src",                # directory where fine-tuning script is stored
    instance_type="ml.g5.xlarge",          # instance type
    instance_count=1,                       # number of instances
    role=ROLE,                              # IAM role used in training job to acccess AWS resources (S3)
    transformers_version="4.28",             # Transformers version
    pytorch_version="2.0",                  # PyTorch version
    py_version="py310",                      # Python version
    checkpoint_s3_uri=OUTPUT_DIR,
    #volume_size=100,
    output_path=OUTPUT_DIR,
    use_spot_instances=True,
    max_wait=30000, # This should be equal to or greater than max_run in seconds'
    max_run=25000,
    base_job_name='gpt-2-mlflow',
    subnets=[subnet],
    security_group_ids=[security_group_id],
    hyperparameters=hyperparameters,         # hyperparameters to use in training job
)

huggingface_estimator.fit({"train": TRAIN_PATH, "test": VALIDATION_PATH})