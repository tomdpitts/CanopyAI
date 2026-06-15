import os, sys, json, glob, argparse; os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_DATASETS_OFFLINE']='1'
import numpy as np, cv2
from datasets import load_dataset
ap=argparse.ArgumentParser(); ap.add_argument('--dirs',required=True,help='comma-separated pred-dir paths under repo'); ap.add_argument('--limit',type=int,default=0); a=ap.parse_args()
# image_id -> stem
id2stem={}
for mp in glob.glob('data/tcd/images/data/tcd/val/*_meta.json'):
    mt=json.load(open(mp)); id2stem[mt['image_id']]=mp.split('/')[-1].replace('_meta.json','')
def crown_mask(geojson,H,W):
    msk=np.zeros((H,W),np.uint8)
    if not os.path.exists(geojson): return None
    try: gj=json.load(open(geojson))
    except: return msk
    for ft in gj.get('features',[]):
        g=ft.get('geometry') or {}
        if g.get('type')=='Polygon': rings=[g['coordinates']]
        elif g.get('type')=='MultiPolygon': rings=g['coordinates']
        else: continue
        for poly in rings:
            pts=np.array(poly[0]).round().astype(np.int32)
            if len(pts)>=3: cv2.fillPoly(msk,[pts],1)
    return msk
ds=load_dataset('restor/tcd',split='test'); n=len(ds) if not a.limit else min(a.limit,len(ds))
dirs=a.dirs.split(',')
acc={d:[0,0,0,0] for d in dirs}  # tp,fp,fn, n_missing
for i in range(n):
    ex=ds[i]; iid=ex['image_id']; stem=id2stem.get(iid)
    gt=(np.array(ex['annotation'].convert('L'))>0); H,W=gt.shape
    for d in dirs:
        cm=crown_mask(f'{d}/{stem}_canopyai.geojson',H,W) if stem else None
        if cm is None: acc[d][3]+=1; continue
        pred=cm>0
        acc[d][0]+=int((pred&gt).sum()); acc[d][1]+=int((pred&~gt).sum()); acc[d][2]+=int((~pred&gt).sum())
print(f'{"pred_dir":>42} {"P":>7}{"R":>7}{"areaF1":>8} {"missing":>8}')
for d in dirs:
    tp,fp,fn,miss=acc[d]; p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f1=2*tp/(2*tp+fp+fn+1e-9)
    print(f'{d.split("/")[-1]:>42} {p:>7.4f}{r:>7.4f}{f1:>8.4f} {miss:>8}')
