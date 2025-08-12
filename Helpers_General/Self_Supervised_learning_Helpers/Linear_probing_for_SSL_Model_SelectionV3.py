from pathlib import Path
from datetime import datetime
import shutil
import tempfile
import torch
import mlflow
import pandas as pd

from rfdetr import RFDETRBase
from Utils_MLFLOW import setup_mlflow_experiment


def wrap_ssl_encoder_for_rfdetr(encoder_path, num_classes=5):
    encoder_weights = torch.load(encoder_path, map_location="cpu")
    raw_weights = encoder_weights["model"] if "model" in encoder_weights else encoder_weights

    wrapped_weights = {f"backbone.{k}": v for k, v in raw_weights.items()}
    wrapped_weights["class_embed.weight"] = torch.zeros(num_classes, 256)
    wrapped_weights["class_embed.bias"] = torch.zeros(num_classes)

    checkpoint = {"model": wrapped_weights}
    wrapped_path = encoder_path.replace(".pt", "_converted_rfdetr_checkpoint.pth")
    torch.save(checkpoint, wrapped_path)

    return wrapped_path


def build_fake_dataset_dir(img_dir, train_json_path, val_json_path):
    tmp_dir = Path(tempfile.mkdtemp())

    train_dir = tmp_dir / "Train"
    val_dir = tmp_dir / "Valid"
    train_images = train_dir / "images"
    val_images = val_dir / "images"

    train_images.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)

    for img_file in Path(img_dir).glob("*.jpg"):
        shutil.copy(img_file, train_images / img_file.name)
        shutil.copy(img_file, val_images / img_file.name)

    shutil.copy(train_json_path, train_dir / "_annotations.coco.json")
    shutil.copy(val_json_path, val_dir / "_annotations.coco.json")
    shutil.copy(train_json_path, train_dir / "images" / "_annotations.coco.json")
    shutil.copy(val_json_path, val_dir / "images" / "_annotations.coco.json")

    return tmp_dir

def linear_probe_with_rf_detr(checkpoint_path, data_path, mlflow_run_name):
    model = RFDETRLinearProbe(pretrain_weights=checkpoint_path)
    return model.train_linear_probe(
        dataset_dir=data_path,
        run_name=mlflow_run_name,
        epochs=1,
        batch_size=16,
        lr=1e-4
    )

class RFDETRLinearProbe(RFDETRBase):
    def train_linear_probe(self, dataset_dir, run_name, epochs=10, batch_size=16, lr=1e-4):
        mlflow.set_experiment("Linear Probing for SSL Checkpoints")

        # ✅ Correct unwrapping and freezing logic
        if hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
            inner_model = self.model.model
            for name, param in inner_model.named_parameters():
                if name.startswith("backbone."):
                    param.requires_grad = False
            print("✅ Backbone frozen.")
        else:
            raise RuntimeError("Could not access inner model for freezing.")

        config = self.get_train_config(
            dataset_dir=str(dataset_dir),
            epochs=epochs,
            batch_size=batch_size,
            amp=False,
            early_stopping=False,
            onecyclelr=False,
            lr=lr,
            eval_only_at_end=True,
            save_checkpoints=True,
        )

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dataset_dir": str(dataset_dir),
        })

        stats = self.train_from_config(config=config, trackExperiment=False)

        map50 = stats.get("coco_eval_bbox", [None])[0]
        if map50 is None:
            raise ValueError("No valid mAP@50 found during linear probing.")

        mlflow.log_metric("final_map@50", map50)

        return map50

def run_linear_probe_all_with_rfdetr(ssl_data_path, save_folder, current_SSL_RunID):
    img_dir = Path(Path(ssl_data_path).parent) / "Supervised" / "Train" / "images"
    # annotations_train = Path(f"./Checkpoints/{current_SSL_RunID}/pure_patch_class_map_balanced_train.json")
    # annotations_val = Path(f"./Checkpoints/{current_SSL_RunID}/pure_patch_class_map_balanced_val.json")

    annotations_train = Path(save_folder) / "pure_patch_class_map_balanced_train.json"
    annotations_val = Path(save_folder) / "pure_patch_class_map_balanced_val.json"
    annotations_train = Path(f"./Checkpoints/run_20250618_101815/pure_patch_class_map_balanced_train.json")
    annotations_val = Path(f"./Checkpoints/run_20250618_101815/pure_patch_class_map_balanced_val.json")

    # annotations = "./Checkpoints/" + current_SSL_RunID + "/pure_patch_class_map_balanced.json"
    # annotations = "./Checkpoints/" + "run_20250618_101815" + "/pure_patch_class_map_balanced.json"

    checkpoints = sorted(Path("./Checkpoints/run_20250618_101815/").glob("*.pt"))
    #checkpoints = sorted(Path(f"./Checkpoints/{current_SSL_RunID}/").glob("*.pt"))

    synthetic_dataset_dir = build_fake_dataset_dir(img_dir, annotations_train, annotations_val)

    experiment_id = setup_mlflow_experiment("LinearProbing_RFDETR")
    run_name = f"LinearProbing_{current_SSL_RunID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"📂 Probing checkpoint {i}/{len(checkpoints)}: {ckpt.name}")
            wrapped_ckpt_path = wrap_ssl_encoder_for_rfdetr(str(ckpt))
            map50 = linear_probe_with_rf_detr(str(wrapped_ckpt_path), synthetic_dataset_dir, run_name)
            mlflow.log_metric(f"mAP50_{ckpt.stem}", map50)
            results.append({"checkpoint": str(ckpt), "mAP50": map50})

        df = pd.DataFrame(results)
        best_row = df.loc[df["mAP50"].idxmax()]
        best_ckpt = best_row["checkpoint"]
        print(f"✅ Best checkpoint: {best_ckpt} with mAP@50 = {best_row['mAP50']:.4f}")

        Path(save_folder).mkdir(parents=True, exist_ok=True)
        best_path_file = Path(save_folder) / "best_encoder_path.txt"
        best_path_file.write_text(best_ckpt)
        mlflow.log_artifact(str(best_path_file))

        result_csv = Path(save_folder) / "linear_probe_results.csv"
        df.to_csv(result_csv, index=False)
        mlflow.log_artifact(str(result_csv))

        shutil.rmtree(synthetic_dataset_dir)

    return best_ckpt

