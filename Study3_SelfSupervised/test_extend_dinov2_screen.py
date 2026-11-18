"""Check continuation plans and learning-curve preservation without training."""
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from extend_dinov2_screen import STOCK_SHA, plan
from probe_dinov2_teachers import merge_metrics


class ExtensionTests(unittest.TestCase):
    def test_seven_jobs_resume_teachers_but_not_random(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(work=root, source=root / "source", output=root / "out", weights=root / "stock")
            names = [("original20", "ssl_dinov2_small16_scratch/teacher_epoch_020.pth")]
            names += [(f"long{e}", f"ssl_dinov2_small16_scratch_200/teacher_epoch_{e:03d}.pth")
                      for e in (20, 50, 100, 150, 200)]
            for name, relative in names:
                teacher = root / relative
                teacher.parent.mkdir(exist_ok=True)
                teacher.write_bytes(name.encode())
                source = args.source / name
                source.mkdir(parents=True)
                (source / "complete.json").write_text(json.dumps(dict(
                    epochs=10, frozen_encoder=False, detector="large", stock_sha256=STOCK_SHA,
                    teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest())))
                for file in ("last.ckpt", "initialization.pth", "metrics.csv"):
                    (source / file).touch()
            jobs = plan(args)
            self.assertEqual(len(jobs), 7)
            for name, command in jobs:
                self.assertEqual(command[command.index("--epochs") + 1], "40")
                self.assertEqual("--resume" in command, name != "random")
                self.assertEqual("--teacher" in command, name != "random")
            (args.source / "long100/last.ckpt").unlink()
            with self.assertRaises(FileNotFoundError):
                plan(args)

    def test_merge_preserves_all_epochs_and_union_of_columns(self):
        with TemporaryDirectory() as temporary:
            history, current = [Path(temporary) / name for name in ("old.csv", "new.csv")]
            history.write_text("epoch,val/ema_mAP_50_95\n9,0.46\n")
            current.write_text("epoch,val/ema_mAP_50_95,val/loss\n10,0.47,3.0\n")
            merge_metrics(history, current)
            with current.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["epoch"] for r in rows], ["9", "10"])
            self.assertEqual(rows[0]["val/loss"], "")
            self.assertEqual(rows[1]["val/loss"], "3.0")


if __name__ == "__main__":
    unittest.main()
