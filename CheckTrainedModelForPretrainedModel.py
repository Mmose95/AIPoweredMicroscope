import zipfile, io, torch

zip_path = r"C:\Users\SH37YE\Downloads\Conf v1 results Epi and Leu 100% data.zip"
inner = r"session_20260225_143233/SquamousEpithelialCell/HPO_Config_004/checkpoint_best_total.pth"

with zipfile.ZipFile(zip_path, "r") as zf:
    blob = zf.read(inner)

ckpt = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
args = ckpt.get("args")
args = vars(args) if hasattr(args, "__dict__") else dict(args)

for k in ["pretrain_weights", "pretrained_encoder", "encoder", "dataset_dir", "seed"]:
    print(f"{k}: {args.get(k)}")