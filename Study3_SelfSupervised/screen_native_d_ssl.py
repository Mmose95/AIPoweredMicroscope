"""Short frozen detector screens; not a strict linear classification probe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def main(a):
    import torch
    from probe_dinov2_teachers import prepare
    from rfdetr import RFDETRSmall
    count=torch.cuda.device_count()
    if not count:raise RuntimeError('No visible CUDA devices')
    output=a.experiment/'frozen_detector_screen'
    teachers=[(f'epoch{e:03d}',a.experiment/'training'/f'teacher_epoch_{e:03d}.pth') for e in a.checkpoints]
    for _,p in teachers:
        if not p.is_file():raise FileNotFoundError(p)
    teachers.append(('random_backbone',None))
    output.mkdir(parents=True,exist_ok=True)
    dataset=prepare(SimpleNamespace(work=a.work,output=output,full_finetune=False))
    stock=RFDETRSmall();weights=Path(stock.model_config.pretrain_weights).resolve();del stock
    queue=[]
    for label,teacher in teachers:
        dest=output/label
        if (dest/'complete.json').is_file():continue
        if dest.exists():raise FileExistsError(f'Interrupted output; preserve it and choose a new screen location: {dest}')
        queue.append((label,teacher,dest))
    active=[];free=list(range(count));failed=[]
    while queue or active:
        while free and queue:
            gpu=free.pop(0);label,teacher,dest=queue.pop(0)
            command=[sys.executable,'-u',str(Path(__file__).with_name('probe_dinov2_teachers.py')),
                     '--worker','--weights',str(weights),'--dataset',str(dataset),'--output',str(dest),'--epochs',str(a.epochs)]
            if teacher:command+=['--teacher',str(teacher)]
            log=(output/f'{label}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
            active.append((gpu,label,subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT),log))
            print(f'[PROBE] GPU{gpu} {label}: {output/ (label+".log")}',flush=True)
        for item in active[:]:
            gpu,label,process,log=item
            status=process.poll()
            if status is not None:
                log.close();active.remove(item);free.append(gpu)
                if status:failed.append(label)
        if active:time.sleep(5)
    if failed:raise RuntimeError(f'Failed probes: {failed}')
    import csv
    result=[]
    for label,_ in teachers:
        with (output/label/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        values=[float(r['val/ema_mAP_50_95']) for r in rows if r.get('val/ema_mAP_50_95')]
        result.append(dict(checkpoint=label,best_ema_map=max(values),last_ema_map=values[-1]))
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[10,30,50])
    p.add_argument('--epochs',type=int,default=10)
    main(p.parse_args())
