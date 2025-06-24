from detr import RFDETRBase
import tempfile
import shutil
import torch

def wrap_ssl_encoder_for_rfdetr(encoder_path, num_classes=5):
    encoder_weights = torch.load(encoder_path, map_location="cpu")
    raw_weights = encoder_weights["model"] if "model" in encoder_weights else encoder_weights

    wrapped_weights = {f"backbone.{k}": v for k, v in raw_weights.items()}

    # Add dummy detection head weights
    wrapped_weights["class_embed.weight"] = torch.zeros(num_classes, 256)
    wrapped_weights["class_embed.bias"] = torch.zeros(num_classes)

    checkpoint = {"model": wrapped_weights}

    wrapped_path = encoder_path.replace(".pt", "_converted_rfdetr_checkpoint.pth")
    torch.save(checkpoint, wrapped_path)

    return wrapped_path


def build_fake_dataset_dir(img_dir, train_json_path, val_json_path):
    tmp_dir = Path(tempfile.mkdtemp())

    # Create Train and Valid folders
    train_dir = tmp_dir / "Train"
    val_dir = tmp_dir / "Valid"
    train_images = train_dir / "images"
    val_images = val_dir / "images"

    train_images.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)

    # Copy all relevant images to both Train and Valid image folders
    '''Note, yes, its the same (which is all) images are moved to both the train and val folder however, the annotation
    #files are different, meaning that during training time it is the annotation files that determine which images
    are loaded - the actual images folders are just containers for images to select from
    Also, since the data is sampled from data that has already been subject to the "no same samples in different datasets, 
    we are already in the clear in regards to that here.'''

    for img_file in Path(img_dir).glob("*.jpg"):
        shutil.copy(img_file, train_images / img_file.name)
        shutil.copy(img_file, val_images / img_file.name)

    # Copy COCO annotation files
    shutil.copy(train_json_path, train_dir / "_annotations.coco.json")
    shutil.copy(val_json_path, val_dir / "_annotations.coco.json")

    shutil.copy(train_json_path, train_dir / "images" / "_annotations.coco.json")
    shutil.copy(val_json_path, val_dir / "images"/ "_annotations.coco.json")

    return tmp_dir

def linear_probe_with_rf_detr(checkpoint_path, dataset , mlflow_experiment_name):
    model = RFDETRBase(pretrain_weights=checkpoint_path)
    # Force unwrap the real nn.Module from nested structure (only affects this local instance)
    if hasattr(model.model, "model") and isinstance(model.model.model, torch.nn.Module):
        inner_model = model.model.model
        for name, param in inner_model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
        print("✅ Backbone frozen.")
    else:
        raise RuntimeError("Could not access inner model for freezing.")

    #model.freeze_backbone()  # now this works because model.model is an nn.Module

    model.train(
        dataset_dir=str(dataset),
        epochs=10,
        batch_size=16,
        trackExperiment=True,
        amp=False,
        early_stopping=True,
        onecyclelr=False,
        lr=1e-4,
        eval_only_at_end=True,
        save_checkpoints=False
    )
    return model.best_map50


from pathlib import Path
import mlflow
import pandas as pd
from datetime import datetime
from Utils_MLFLOW import setup_mlflow_experiment

def run_linear_probe_all_with_rfdetr(ssl_data_path, save_folder, current_SSL_RunID):

    img_dir = Path(Path(ssl_data_path).parent) / "Supervised" / "Train" / "images"
    #annotations_train = Path(f"./Checkpoints/{current_SSL_RunID}/pure_patch_class_map_balanced_train.json")
    #annotations_val = Path(f"./Checkpoints/{current_SSL_RunID}/pure_patch_class_map_balanced_val.json")

    annotations_train = Path(save_folder) / "pure_patch_class_map_balanced_train.json"
    annotations_val = Path(save_folder) / "pure_patch_class_map_balanced_val.json"
    annotations_train = Path(f"./Checkpoints/run_20250618_101815/pure_patch_class_map_balanced_train.json")
    annotations_val = Path(f"./Checkpoints/run_20250618_101815/pure_patch_class_map_balanced_val.json")

    # annotations = "./Checkpoints/" + current_SSL_RunID + "/pure_patch_class_map_balanced.json"
    #annotations = "./Checkpoints/" + "run_20250618_101815" + "/pure_patch_class_map_balanced.json"

    #checkpoints = sorted(Path(save_folder).glob("*.pt"))
    checkpoints = sorted(Path("./Checkpoints/run_20250618_101815/").glob("*.pt"))

    synthetic_dataset_dir = build_fake_dataset_dir(img_dir, annotations_train, annotations_val) #mind the input index of the datasets

    experiment_id = setup_mlflow_experiment("LinearProbing_RFDETR")
    run_name = f"LinearProbing_{current_SSL_RunID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        for ckpt in checkpoints:
            print(f"📂 Probing checkpoint: {ckpt.name}")
            wrapped_ckpt_path = wrap_ssl_encoder_for_rfdetr(str(ckpt)) #we are only pointing to an encoder, we need to wrap it.
            map50 = linear_probe_with_rf_detr(str(wrapped_ckpt_path), synthetic_dataset_dir, run_name)
            mlflow.log_metric(f"mAP50_{ckpt.stem}", map50)
            results.append({"checkpoint": str(ckpt), "mAP50": map50})

        df = pd.DataFrame(results)
        best_row = df.loc[df["mAP50"].idxmax()]
        best_ckpt = best_row["checkpoint"]
        print(f"✅ Best checkpoint: {best_ckpt} with mAP@50 = {best_row['mAP50']:.4f}")

        # Save best path and CSV for traceability
        Path(ssl_checkpoints_dir).mkdir(parents=True, exist_ok=True)
        best_path_file = Path(ssl_checkpoints_dir) / "best_encoder_path.txt"
        best_path_file.write_text(best_ckpt)
        mlflow.log_artifact(str(best_path_file))

        result_csv = Path(ssl_checkpoints_dir) / "linear_probe_results.csv"
        df.to_csv(result_csv, index=False)
        mlflow.log_artifact(str(result_csv))

        shutil.rmtree(synthetic_dataset_dir) #Clenup the temp dataset dir after probing to not save tons of data

    return best_ckpt
