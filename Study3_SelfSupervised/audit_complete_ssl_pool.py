"""Read-only, resumable audit of EVERY training-pool image; no model inference/training.

Image measurements describe stain/texture, not cell identities. Training annotations
are positive controls; lack of an annotation is never a negative label.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sqlite3
import time

VERSION = 1


def specimen_id(name):
    return int(re.search(r'Sample[ _]*(\d+)',name,re.I).group(1))


def crop_params(width,height,scale,rng):
    """Same geometric distribution as torchvision RandomResizedCrop; new RNG draws."""
    for _ in range(10):
        area=width*height*rng.uniform(*scale)
        aspect=math.exp(rng.uniform(math.log(.75),math.log(4/3)))
        w,h=round(math.sqrt(area*aspect)),round(math.sqrt(area/aspect))
        if 0<w<=width and 0<h<=height:
            return int(rng.integers(width-w+1)),int(rng.integers(height-h+1)),w,h
    ratio=width/height
    if ratio<.75:w=width;h=round(w/.75)
    elif ratio>4/3:h=height;w=round(h*4/3)
    else:w,h=width,height
    return (width-w)//2,(height-h)//2,w,h


def map_annotations(rows,coco):
    images={i['id']:i for i in coco['images']}
    cats={c['id']:c['name'] for c in coco['categories']}
    lookup={}
    for row in rows:
        m=re.search(r'_X(\d+)_Y(\d+)',row['relative_path'])
        if m:lookup[(specimen_id(row['specimen']),*map(int,m.groups()))]=row['relative_path']
    result=defaultdict(list);excluded=Counter();seen=set()
    for a in coco['annotations']:
        im=images[a['image_id']]
        m=re.search(r'Sample_(\d+)_CAL10PATCH_X(\d+)_Y(\d+)_W(\d+)_H(\d+)',im['file_name'])
        if not m:raise ValueError(f'Cannot map annotated image: {im}')
        sid,x0,y0,w,h=map(int,m.groups())
        if (im['width'],im['height'])!=(w,h):raise ValueError('Annotation image was resized')
        x,y,bw,bh=a['bbox']
        if x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1:
            excluded['annotation_boundary']+=1;continue
        x+=x0;y+=y0
        key=(sid,round(x,2),round(y,2),round(bw,2),round(bh,2),a['category_id'])
        if key in seen:excluded['duplicate_box']+=1;continue
        seen.add(key)
        tx,ty=int((x+bw/2)//2584)*2584,int((y+bh/2)//1936)*1936
        path=lookup.get((sid,tx,ty))
        if not path:excluded['not_in_pool']+=1;continue
        if x<tx or y<ty or x+bw>tx+2584 or y+bh>ty+1936:
            excluded['original_tile_boundary']+=1;continue
        result[path].append(dict(x=x-tx,y=y-ty,w=bw,h=bh,category=cats[a['category_id']],annotation_id=a['id']))
    return dict(result),dict(excluded)


def measure(item):
    import cv2
    import numpy as np
    from PIL import Image
    row,root,annotations=item
    path=root/row['relative_path']
    record=dict(relative_path=row['relative_path'],specimen=row['specimen'],specimen_id=specimen_id(row['specimen']))
    try:
        data=path.read_bytes()
        record.update(bytes=len(data),expected_bytes=int(row['size_bytes']),
                      size_matches_manifest=len(data)==int(row['size_bytes']),sha256=hashlib.sha256(data).hexdigest())
        with Image.open(BytesIO(data)) as header:
            width,height=header.size;record.update(width=width,height=height,format=header.format)
        if record['format']=='JPEG' and not data.rstrip().endswith(b'\xff\xd9'):
            raise ValueError('JPEG does not end with EOI marker')
        thumb=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_REDUCED_COLOR_4)
        if thumb is None:raise ValueError('Reduced image decoder returned no image')
        if max(thumb.shape[:2])>700:
            scale=646/max(thumb.shape[:2]);thumb=cv2.resize(thumb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(thumb,cv2.COLOR_BGR2GRAY)
        hsv=cv2.cvtColor(thumb,cv2.COLOR_BGR2HSV);sat=hsv[...,1]
        # Deliberately separate proxies; none is described as a biological cell mask.
        stained=((sat>=24)&(gray<245))|(gray<175)
        weak=(sat>=12)&(gray<250)
        dark=gray<190
        grad=cv2.Laplacian(gray,cv2.CV_32F)
        hist=np.bincount((gray//8).ravel(),minlength=32).astype(float);hist/=hist.sum()
        entropy=float(-(hist[hist>0]*np.log2(hist[hist>0])).sum())
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
        count,_,stats,_=cv2.connectedComponentsWithStats(stained.astype(np.uint8))
        areas=stats[1:,cv2.CC_STAT_AREA]
        # Blocks describe spatial concentration, without asserting cell identity.
        block=np.array([stained[a:b,c:d].mean() for a,b in zip(np.linspace(0,gray.shape[0],9,dtype=int)[:-1],np.linspace(0,gray.shape[0],9,dtype=int)[1:])
                        for c,d in zip(np.linspace(0,gray.shape[1],9,dtype=int)[:-1],np.linspace(0,gray.shape[1],9,dtype=int)[1:])])
        digest=cv2.resize(gray,(9,8));dhash=np.packbits(digest[:,1:]>digest[:,:-1]).tobytes().hex()
        record.update(status='ok',thumb_width=gray.shape[1],thumb_height=gray.shape[0],
                      luminance_mean=float(gray.mean()),luminance_std=float(gray.std()),
                      saturation_mean=float(sat.mean()),weak_stain_fraction=float(weak.mean()),
                      stain_fraction=float(stained.mean()),dark_fraction=float(dark.mean()),
                      bright_fraction=float((gray>=245).mean()),entropy=entropy,
                      laplacian_variance=float(grad.var()),tenengrad=float((gx*gx+gy*gy).mean()),
                      components_ge5=int((areas>=5).sum()),components_ge30=int((areas>=30).sum()),
                      components_ge100=int((areas>=100).sum()),max_block_stain=float(block.max()),
                      stain_blocks_ge1percent=int((block>=.01).sum()),dhash=dhash,
                      known_annotations=len(annotations),known_leucocytes=sum(a['category']=='Leucocyte' for a in annotations),
                      known_secs=sum(a['category']=='Squamous Epithelial Cell' for a in annotations))
        positive=[]
        sx,sy=gray.shape[1]/width,gray.shape[0]/height
        for a in annotations:
            x1,y1=int(a['x']*sx),int(a['y']*sy)
            x2,y2=max(x1+1,math.ceil((a['x']+a['w'])*sx)),max(y1+1,math.ceil((a['y']+a['h'])*sy))
            roi=stained[y1:y2,x1:x2]
            positive.append({**a,'stain_fraction_in_box':float(roi.mean()),'tile':row['relative_path']})
        rng=np.random.default_rng(int(record['sha256'][:16],16))
        integral=cv2.integral(stained.astype(np.uint8))
        for kind,number,scale in [('global',2,(.32,1)),('local',8,(.05,.32))]:
            fractions=[]
            for _ in range(number):
                x,y,w,h=crop_params(width,height,scale,rng)
                x1,y1=int(x*sx),int(y*sy);x2,y2=max(x1+1,round((x+w)*sx)),max(y1+1,round((y+h)*sy))
                x2=min(x2,gray.shape[1]);y2=min(y2,gray.shape[0])
                total=integral[y2,x2]-integral[y1,x2]-integral[y2,x1]+integral[y1,x1]
                fractions.append(float(total/((x2-x1)*(y2-y1))))
            record[kind+'_stain_fractions']=fractions
        return record,positive
    except Exception as exc:
        record.update(status='error',error=f'{type(exc).__name__}: {exc}')
        return record,[]


def main(args):
    import cv2
    cv2.setNumThreads(1)
    with args.manifest.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    split_ids={}
    for split in ('train','valid','test'):
        coco=json.loads((args.dataset/split/'_annotations.coco.json').read_text())
        split_ids[split]={specimen_id(i['file_name']) for i in coco['images']}
    if any(split_ids[a]&split_ids[b] for a,b in [('train','valid'),('train','test'),('valid','test')]):
        raise ValueError('Specimen overlap in supervised split')
    pool_ids={specimen_id(r['specimen']) for r in rows}
    if pool_ids!=split_ids['train']:raise ValueError('Pool does not match all and only training specimens')
    if len({r['relative_path'] for r in rows})!=len(rows):raise ValueError('Duplicate manifest paths')
    controls,excluded=map_annotations(rows,json.loads((args.dataset/'train/_annotations.coco.json').read_text()))
    args.output.mkdir(parents=True,exist_ok=True)
    meta=dict(version=VERSION,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              coco_sha256=hashlib.sha256((args.dataset/'train/_annotations.coco.json').read_bytes()).hexdigest(),
              image_root=str(args.images.resolve()),images=len(rows),training_specimens=sorted(pool_ids),
              excluded_validation_specimens=sorted(split_ids['valid']),excluded_test_specimens=sorted(split_ids['test']),
              annotated_tiles=len(controls),mapped_annotations=sum(map(len,controls.values())),annotation_exclusions=excluded,
              measurement='Reduced JPEG decoding (usually quarter dimensions); original dimensions/EOI checked, exact file hashes; no native-resolution focus certification',
              cell_interpretation='Stain, texture and connected components are proxies, not cell labels; unannotated is unknown')
    metadata=args.output/'audit_metadata.json'
    if metadata.exists() and json.loads(metadata.read_text())!=meta:raise ValueError('Audit inputs/config changed; use a fresh output')
    metadata.write_text(json.dumps(meta,indent=2))
    (args.output/'annotation_positive_controls.json').write_text(json.dumps(controls))
    db=sqlite3.connect(args.output/'pool_audit.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS images(path TEXT PRIMARY KEY, specimen TEXT, status TEXT, record TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS positives(path TEXT PRIMARY KEY, record TEXT)')
    done={r[0] for r in db.execute('SELECT path FROM images')}
    missing=[r for r in rows if r['relative_path'] not in done]
    # Verify the disk inventory too; don't silently assume the historical manifest is complete.
    expected={r['relative_path'] for r in rows};actual=set()
    for folder in args.images.iterdir():
        if folder.is_dir() and re.match(r'^sample[ _]*\d',folder.name,re.I) and specimen_id(folder.name) in pool_ids:
            actual.update(p.relative_to(args.images).as_posix() for p in folder.iterdir()
                          if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'))
    inventory=dict(manifest_images=len(expected),disk_images=len(actual),missing=sorted(expected-actual),extra=sorted(actual-expected))
    (args.output/'inventory_reconciliation.json').write_text(json.dumps(inventory,indent=2))
    print(f'Auditing ALL {len(rows):,} images from {len(pool_ids)} training specimens; {len(done):,} already audited. Disk inventory {len(actual):,}.',flush=True)
    start=time.monotonic();completed=0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0,len(missing),1000):
            batch=missing[offset:offset+1000]
            for record,positive in pool.map(measure,((r,args.images,controls.get(r['relative_path'],[])) for r in batch)):
                db.execute('INSERT INTO images VALUES(?,?,?,?)',(record['relative_path'],record['specimen'],record['status'],json.dumps(record)))
                if positive:db.execute('INSERT INTO positives VALUES(?,?)',(record['relative_path'],json.dumps(positive)))
                completed+=1
            db.commit()
            elapsed=time.monotonic()-start;rate=completed/max(elapsed,.001)
            print(f'[AUDIT] {len(done)+completed:,}/{len(rows):,}; {rate:.1f} images/s; remaining {(len(missing)-completed)/max(rate,.001)/60:.1f} min',flush=True)
    counts=dict(db.execute('SELECT status,COUNT(*) FROM images GROUP BY status'))
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    (args.output/'audit_complete.json').write_text(json.dumps(dict(total=len(rows),counts=counts,inventory=inventory,
                                                               elapsed_this_invocation_seconds=time.monotonic()-start),indent=2))
    print('COMPLETE',counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('images','manifest','dataset','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    main(parser.parse_args())
