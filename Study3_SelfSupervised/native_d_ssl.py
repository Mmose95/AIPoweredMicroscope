"""Native 768/384 D recipe and reproducible weighted distributed sampling."""
import csv
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision.transforms import functional as F
from ssl_morphology_review import choose_anchor, choose_windows


class NativeViews:
    def __call__(self, image):
        import cv2
        cv2.setNumThreads(1)
        rng = np.random.default_rng(random.getrandbits(64))
        anchor = choose_anchor(np.asarray(image), rng, component_balanced=True)
        geometry = choose_windows(*image.size, rng, anchor, global_size=768,
                                  local_size=384, region_size=1024)
        def view(box):
            x,y,w,h=box
            crop=image.crop((x,y,x+w,y+h))
            if rng.random()<.5: crop=F.hflip(crop)
            if rng.random()<.5: crop=F.vflip(crop)
            crop=crop.rotate(int(rng.integers(4))*90)
            crop=F.adjust_brightness(crop,float(rng.uniform(.9,1.1)))
            crop=F.adjust_contrast(crop,float(rng.uniform(.9,1.1)))
            crop=F.adjust_saturation(crop,float(rng.uniform(.95,1.05)))
            return F.normalize(F.to_tensor(crop),(.485,.456,.406),(.229,.224,.225))
        globals_=[view(b) for b in geometry['global_boxes']]
        return dict(global_crops=globals_,global_crops_teacher=globals_,
                    local_crops=[view(b) for b in geometry['local_boxes']],offsets=())


class WeightedDistributedDraws(Sampler):
    """Every rank slices the SAME seeded global draw; replacement is intentional."""
    def __init__(self, weights, batch_size, world, rank):
        self.weights=torch.as_tensor(weights,dtype=torch.double)
        self.world,self.rank=world,rank
        self.total=(len(weights)//batch_size)*batch_size
        if self.total<batch_size: raise ValueError('Pool is smaller than one global batch')
        self.epoch=0
    def set_epoch(self, epoch): self.epoch=epoch
    def __len__(self): return self.total//self.world
    def __iter__(self):
        draws=torch.multinomial(self.weights,self.total,replacement=True,
                               generator=torch.Generator().manual_seed(self.epoch))
        return iter(draws[self.rank::self.world].tolist())


def load_weights(path, paths, root):
    with path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    mapping={r['relative_path']:float(r['probability']) for r in rows}
    relative=[p.relative_to(root).as_posix() for p in paths]
    if len(mapping)!=len(rows) or set(mapping)!=set(relative):
        raise ValueError('Sampling inventory must exactly match the training manifest')
    weights=[mapping[p] for p in relative]
    if not all(np.isfinite(w) and w>=0 for w in weights) or abs(sum(weights)-1)>1e-6:
        raise ValueError('Invalid probability weights')
    return weights


def prepare(manifest, images, output, workers=8):
    """Recompute audited coarse metrics once on UCloud; no annotations gate images."""
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from audit_complete_ssl_pool import measure
    import cv2
    cv2.setNumThreads(1)
    with manifest.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    provenance=dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    images=str(images.resolve()),recipe='native_D_768_384_v1')
    metadata=output.with_suffix('.json')
    if output.exists():
        if json.loads(metadata.read_text())!=provenance:raise ValueError('Existing weights have different provenance')
        print('Reusing verified sampling inventory:',output,flush=True);return
    grouped=defaultdict(list)
    scores={'varied':1.,'sparse':.5,'low':.25,'very_low':.05,'black':0.}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(record,_) in enumerate(pool.map(measure,((r,images,[]) for r in rows)),1):
            if record['status']!='ok':raise ValueError(f'Image audit failed: {record}')
            b='varied'
            if record['luminance_mean']<5 and record['luminance_std']<2:b='black'
            elif record['stain_fraction']<.001 and record['max_block_stain']<.01 and record['luminance_std']<5:b='very_low'
            elif record['luminance_std']<6:b='low'
            elif record['stain_fraction']<.01:b='sparse'
            grouped[record['specimen']].append((record['relative_path'],b))
            if i%1000==0:print(f'[SAMPLING AUDIT] {i}/{len(rows)}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.part')
    with temporary.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['relative_path','bucket','probability'])
        for specimen,items in sorted(grouped.items()):
            eligible=sum(scores[b]>0 for _,b in items);total=sum(scores[b] for _,b in items)
            if not eligible:raise ValueError(f'No non-black images in {specimen}')
            for name,b in items:
                probability=(.2/eligible+.8*scores[b]/total)/len(grouped) if scores[b] else 0.
                writer.writerow([name,b,probability])
    temporary.replace(output)
    metadata.write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for name in ['manifest','images','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare(a.manifest,a.images,a.output,a.workers)
