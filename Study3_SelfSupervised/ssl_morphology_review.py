"""Shared native view geometry; training uses it only via explicit native-D opt-in."""
import numpy as np


GLOBAL_SIZE=512
LOCAL_SIZE=224
REGION_SIZE=768


def structure_mask(rgb):
    """Local dark structure, not a cell detector; suppress uniform pink and black."""
    import cv2
    h,w=rgb.shape[:2]
    scale=min(1.,646/max(w,h))
    small=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    background=cv2.GaussianBlur(gray.astype(np.float32),(0,0),5)
    mask=((background-gray)>6).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    keep=np.zeros(n,np.uint8)
    if n>1:keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>=5)
    return keep[labels].astype(bool)


def choose_windows(width,height,rng,anchor=None,global_size=GLOBAL_SIZE,local_size=LOCAL_SIZE,region_size=REGION_SIZE):
    """Native-resolution windows; every local is physically inside BOTH globals.

    Uniform and guided sampling share exactly this geometry. Guided only supplies
    a spatial anchor, with jitter; no cell is forced to the image centre.
    """
    if width<region_size or height<region_size:
        raise ValueError('Source too small for the proposed region; no padding/upscaling fallback')
    maximum=region_size-global_size
    shift=min(maximum//2,global_size//4)
    if maximum<0 or local_size>global_size-shift:
        raise ValueError('Local windows do not fit the guaranteed global intersection')
    if anchor is None:
        rx=int(rng.integers(width-region_size+1));ry=int(rng.integers(height-region_size+1))
    else:
        ax,ay=anchor
        rx=int(np.clip(round(ax-region_size/2+rng.uniform(-shift,shift)),0,width-region_size))
        ry=int(np.clip(round(ay-region_size/2+rng.uniform(-shift,shift)),0,height-region_size))
    x1=int(rng.integers(maximum+1));y1=int(rng.integers(maximum+1))
    x2=int(rng.integers(max(0,x1-shift),min(maximum,x1+shift)+1))
    y2=int(rng.integers(max(0,y1-shift),min(maximum,y1+shift)+1))
    globals_=[(rx+x1,ry+y1,global_size,global_size),(rx+x2,ry+y2,global_size,global_size)]
    ix,iy=rx+max(x1,x2),ry+max(y1,y2)
    iw,ih=global_size-abs(x1-x2),global_size-abs(y1-y2)
    locals_=[(int(rng.integers(ix,ix+iw-local_size+1)),int(rng.integers(iy,iy+ih-local_size+1)),local_size,local_size) for _ in range(8)]
    overlap=iw*ih
    return dict(region=[rx,ry,region_size,region_size],global_boxes=globals_,local_boxes=locals_,
                global_intersection_over_union=overlap/(2*global_size**2-overlap),
                resize_native=False,token_patch_size=16,
                global_token_grid=[global_size//16]*2,local_token_grid=[local_size//16]*2)


def component_centres(mask):
    import cv2
    _,_,_,centres=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return centres[1:]


def choose_anchor(rgb,rng,source_size=None,component_balanced=False):
    mask=structure_mask(rgb)
    if component_balanced:
        centres=component_centres(mask)
        if not len(centres):return None
        x,y=centres[int(rng.integers(len(centres)))];width,height=source_size or (rgb.shape[1],rgb.shape[0])
        return (float(x*width/mask.shape[1]),float(y*height/mask.shape[0]))
    y,x=np.where(mask)
    if not len(x):return None
    i=int(rng.integers(len(x)))
    width,height=source_size or (rgb.shape[1],rgb.shape[0])
    return (float((x[i]+.5)*width/mask.shape[1]),float((y[i]+.5)*height/mask.shape[0]))


def inside(box,view):
    x,y,w,h=box;vx,vy,vw,vh=view
    return x>=vx and y>=vy and x+w<=vx+vw and y+h<=vy+vh
