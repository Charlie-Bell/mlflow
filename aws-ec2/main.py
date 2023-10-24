import dotenv
import os

import mlflow
import datetime
import sys
import logging

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

dotenv.load_dotenv()


# Define variables
tracking_uri = os.environ.get("TRACKING_URI")
role = os.environ.get("ROLE")
s3_dir = os.environ.get("S3_DIR")
experiment_name = "experiment-1"

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set remote mlflow server
mlflow.set_tracking_uri(tracking_uri)

# Set experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
if not experiment:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Start run
with mlflow.start_run() as run:
    # Configure hyperparameters
    N_ESTIMATORS = 100
    MAX_DEPTH = 5
    MAX_FEATURES = 2

    # Set run name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%M%H%S")
    RUN_NAME = f"rf-est{N_ESTIMATORS}-dep{MAX_DEPTH}-fea{MAX_FEATURES}-{timestamp}"
    mlflow.set_tag("mlflow.runName", RUN_NAME)

    # Log hyperparameters
    mlflow.log_params({
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "max_features": MAX_FEATURES,
    })

    # Load dataset
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Fit regressor
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, max_features=MAX_FEATURES)
    rf.fit(X_train, y_train)

    # Evaluate predictions
    predictions = rf.predict(X_test)
    squared_error = mean_squared_error(y_test, predictions)

    # Log metrics
    mlflow.log_metric("squared_error", squared_error)

    # Log model + environment to filestore
    signature = mlflow.models.infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, "model", signature=signature)
