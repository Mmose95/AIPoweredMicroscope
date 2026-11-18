"""Multi-cell Cell-DINO views using the released field-of-view augmentation code.

Nuclear connected components are diagnostic proxies, not cell ground truth.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import types


def load_author_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main(args):
    import cv2
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from torchvision.transforms import RandomResizedCrop
    import yaml

    torch.set_num_threads(4)
    config=yaml.safe_load((args.sources/'vitl16_hpafov.yaml').read_text())
    package=types.ModuleType('celldino_reference');package.__path__=[str(args.sources)]
    sys.modules[package.__name__]=package
    tr=load_author_module('celldino_reference.transforms',args.sources/'cell_transforms.py')
    aug=load_author_module('celldino_reference.augmentations',args.sources/'cell_augmentations.py')
    crops=config['crops']
    transform=aug.CellAugmentationDINO(crops['global_crops_scale'],crops['local_crops_scale'],
                                       crops['local_crops_number'],crops['global_crops_size'],crops['local_crops_size'])
    args.output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    inventory=[];measurements={}
    original_params=RandomResizedCrop.get_params
    original_norm=tr.SelfNormalizeNoDiv.forward
    for index,path in enumerate(sorted((args.sources/'fov_samples').glob('*.tiff')),1):
        with Image.open(path) as im:array=np.array(im)
        assert array.shape==(512,512,4)
        folder=args.output/f'field_{index:02d}';folder.mkdir()
        shutil.copy2(path,folder/'original_4channel.tiff')
        original=Image.fromarray(array[...,:3]);original.save(folder/'original_display.png')
        tensor=torch.tensor(array,dtype=torch.float32).permute(2,0,1)
        boxes=[];pre_norm=[]

        def params(img,scale,ratio):
            p=original_params(img,scale,ratio);boxes.append(list(p));return p

        def norm(self,img):
            pre_norm.append(img.clone());return original_norm(self,img)

        seed=20261118+index
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        RandomResizedCrop.get_params=staticmethod(params);tr.SelfNormalizeNoDiv.forward=norm
        try:views=transform(tensor)
        finally:
            RandomResizedCrop.get_params=staticmethod(original_params);tr.SelfNormalizeNoDiv.forward=original_norm
        torch.save(views,folder/'normalized_model_inputs.pt')
        assert len(pre_norm)==10 and len(boxes)==10
        sheet=Image.new('RGB',(1650,940),'white');draw=ImageDraw.Draw(sheet)
        draw.text((20,15),f'MULTI-CELL FIELD {index:02d}: {path.name}',font=font,fill='black')
        draw.text((20,45),'Released 512px field + actual author HPA-FoV augmentations; RGB display shows 3 of 4 channels.',font=font,fill='black')
        sheet.paste(original,(20,100))
        for j,v in enumerate(pre_norm):
            pixels=(v[:3].clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
            view=Image.fromarray(pixels)
            kind,n=('global',j+1) if j<2 else ('local',j-1)
            view.save(folder/f'{kind}_{n:02d}.png')
            grid=view.copy();gd=ImageDraw.Draw(grid)
            for t in range(0,grid.width,16):gd.line((t,0,t,grid.height),fill=(255,220,0))
            for t in range(0,grid.height,16):gd.line((0,t,grid.width,t),fill=(255,220,0))
            grid.save(folder/f'{kind}_{n:02d}_token_grid.png')
            if j<2:x,y=580+j*330,110
            else:x,y=580+(j-2)%4*190,450+(j-2)//4*165
            sheet.paste(view,(x,y));draw.text((x,y+view.height+6),f'{kind} {n}',font=font,fill='black')
        draw.text((20,825),'Global: 224 x 224 (14 x 14 tokens). Local: 96 x 96 (6 x 6 tokens). Token patch: 16 px.',font=font,fill='black')
        draw.text((20,860),'Normalization reversed for display. New random draws, not historical training inputs.',font=font,fill='black')
        sheet.save(folder/'contact_sheet.png')
        inventory.append(dict(field=index,filename=path.name,seed=seed,source_size=[512,512],crop_boxes_top_left_height_width=boxes))

        # Simple reproducible nuclear candidates for scale illustration only.
        smooth=cv2.GaussianBlur(array[...,2],(3,3),0)
        threshold,mask=cv2.threshold(smooth,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        candidates=[]
        qc=original.copy();qd=ImageDraw.Draw(qc)
        for x,y,w,h,area in stats[1:]:
            if area<40 or area>12000 or min(w,h)<4 or max(w,h)/min(w,h)>3:continue
            if x<=0 or y<=0 or x+w>=512 or y+h>=512:continue
            candidates.append((int(x),int(y),int(w),int(h)))
            qd.rectangle((x,y,x+w,y+h),outline=(255,255,0),width=1)
        qc.save(folder/'nuclear_candidate_QC.png')
        inventory[-1]['nuclear_candidates']=candidates
        inventory[-1]['otsu_threshold']=float(threshold)
        # Sensitivity scenarios because the paper reports 256px working images,
        # whereas released fields are 512px and SSL loader uses another directory.
        for side,label in [(512,'released512'),(256,'working256_sensitivity')]:
            factor=side/512
            for x,y,w,h in candidates:
                measurements.setdefault((label,'Native'),[]).append((w*factor,h*factor))
            dummy=Image.new('RGB',(side,side))
            for repeat in range(100):
                for kind,size,scale,count in [('Global',224,(.4,1.),2),('Local',96,(.005,.4),8)]:
                    for _ in range(count):
                        top,left,ch,cw=original_params(dummy,scale,(3/4,4/3))
                        for x,y,w,h in candidates:
                            x,y,w,h=x*factor,y*factor,w*factor,h*factor
                            if x>=left and y>=top and x+w<=left+cw and y+h<=top+ch:
                                measurements.setdefault((label,kind),[]).append((w*size/cw,h*size/ch))
        print(f'[{index}/25] {path.name}: {len(candidates)} nuclear candidates',flush=True)
    summary={}
    for (scenario,kind),values in measurements.items():
        a=np.array(values)
        summary.setdefault(scenario,{})[kind]=dict(appearances=len(values),
            median_width_height_px=np.median(a,axis=0).tolist(),
            equivalent_token_width_p10_p50_p90=(np.percentile(np.sqrt(a[:,0]*a[:,1]),[10,50,90])/16).tolist())
    result=dict(status='Author augmentation reproduction on released multi-cell FoVs, not exact historical SSL dataset reproduction',
                segmentation='Blue-channel Otsu connected components, not annotated nuclei or whole-cell masks; QC overlays supplied',
                source_sample='First 25 image members of public Master_HPA_FOV_512 archive; convenience sample',
                scenarios='released512 measured geometry; working256 sensitivity assumes prior resize to paper-reported working size, not verified historical preprocessing',
                summary=summary,inventory=inventory)
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2))
    page=['<html><meta charset="utf-8"><title>Multi-cell Cell-DINO comparison</title><body>',
          '<h1>Multi-cell Cell-DINO field-of-view comparison</h1>',
          '<p>25 released multi-cell fields, author HPA-FoV augmentation code. Not the single-cell dataset.</p>',
          '<p><a href="REPORT.md">Read methodology and limitations</a></p>']
    for item in inventory:
        f=f"field_{item['field']:02d}"
        page.append(f'<h2>{f}: {item["filename"]}</h2><a href="{f}/original_display.png">Original</a> | '
                    f'<a href="{f}/global_01_token_grid.png">Global token grid</a> | '
                    f'<a href="{f}/local_01_token_grid.png">Local token grid</a> | '
                    f'<a href="{f}/nuclear_candidate_QC.png">Nuclear estimate QC</a><br>'
                    f'<img src="{f}/contact_sheet.png" style="max-width:100%">')
    (args.output/'index.html').write_text('\n'.join(page)+'</body></html>',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
