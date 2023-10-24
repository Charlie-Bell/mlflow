import mlflow
import datetime
import optuna

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def run(trial):
    # Start run
    with mlflow.start_run():
        # Configure hyperparameters
        N_ESTIMATORS = trial.suggest_int("n_estimators", 50, 500)
        MAX_DEPTH = trial.suggest_int("max_depth", 3, 8)
        MAX_FEATURES = trial.suggest_int("max_features", 2, 6)

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

        return squared_error
    
    
if __name__ == "__main__":
    # Create experiment
    EXPERIMENT_NAME = "experiment-1"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        mlflow.create_experiment(EXPERIMENT_NAME)

    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Optuna create study
    study = optuna.create_study()
    study.optimize(run, n_trials=15, timeout=60)

    