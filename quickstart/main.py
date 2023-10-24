import mlflow
import datetime

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from datasets import load_from_disk


if __name__ == "__main__":
    # Create experiment
    EXPERIMENT_NAME = "experiment-1"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        mlflow.create_experiment(EXPERIMENT_NAME)

    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start run
    with mlflow.start_run() as run:
        # Configure hyperparameters
        N_ESTIMATORS = 100
        MAX_DEPTH = 6
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

        # Log model + environment
        signature = mlflow.models.infer_signature(X_test, predictions)
        mlflow.sklearn.log_model(rf, "model", signature=signature)
