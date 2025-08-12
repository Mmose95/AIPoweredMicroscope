# common_utils.py
'''
import mlflow

def setup_mlflow_experiment(experiment_name):
    """Sets up MLflow with the specified experiment name and returns its ID."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Get or create the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        print("Experiment already exists, new runs is placed onto the existing experiment")
        experiment_id = experiment.experiment_id

    return experiment_id
'''


import os
import pathlib
import platform
from urllib.parse import urlparse
import mlflow

def _ensure_dir(path: str) -> str:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()

def _on_ucloud() -> bool:
    return os.path.exists("/work") or "KUBERNETES_SERVICE_HOST" in os.environ

def _default_tracking_uri() -> str:
    # If explicitly set, respect it
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    # Local Windows desktop: keep your server on port 5000
    if platform.system() == "Windows":
        return "http://127.0.0.1:5000"

    # UCloud/Linux: file store
    if _on_ucloud() and os.path.exists("/work"):
        return f"file:{_ensure_dir('/work/CondaEnv/mlflow/mlruns')}"
    return f"file:{_ensure_dir('./mlflows')}"  # local fallback (note: separate folder to avoid old metadata)

def setup_mlflow_experiment(experiment_name: str) -> str:
    tracking_uri = _default_tracking_uri()
    # Defensive: an env MLFLOW_ARTIFACT_URI can force mlflow-artifacts. Clear it for file tracking.
    if tracking_uri.startswith("file:"):
        os.environ.pop("MLFLOW_ARTIFACT_URI", None)

    mlflow.set_tracking_uri(tracking_uri)
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        # If using a file store, ensure artifact_location is also file:
        parsed = urlparse(tracking_uri)
        if parsed.scheme == "file":
            base = parsed.path.rstrip("/")
            art_dir = _ensure_dir(os.path.join(base, "artifacts", exp_name.replace(" ", "_")))
            experiment_id = mlflow.create_experiment(exp_name, artifact_location=f"file:{art_dir}")
        else:
            experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(exp_name)
    print(f"[MLflow] tracking_uri={tracking_uri}  experiment='{exp_name}' (id={experiment_id})")
    return experiment_id

