import sys, json, glob; import numpy as np, cv2, torch
sys.path.insert(0,'phase30'); sys.path.insert(0,'shadow_semseg'); sys.path.insert(0,'.')
import benchmark as B
from config import Config; from model import build_model
import pycocotools.mask as mu
dev='mps' if torch.backends.mps.is_available() else 'cpu'
ck=torch.load('/tmp/shadow_semseg/runs/canopyreg_p22/best.pt',map_location=dev)
cfg=Config(**{k:v for k,v in ck['cfg'].items() if k in Config.__dataclass_fields__})
m=build_model(cfg,verbose=False).to(dev).eval(); m.load_state_dict(ck['model'])
print('epoch',ck.get('epoch'),'val',round(ck.get('val_tree_F1',0),4),'target',cfg.target)
mean=torch.tensor(cfg.imagenet_mean).view(3,1,1); std=torch.tensor(cfg.imagenet_std).view(3,1,1)
import rasterio
metas=sorted(glob.glob('data/tcd/images/data/tcd/val/*_meta.json'))[:15]
f1s=[]; ncc=[]; ngt=[]; covfrac=[]
for mp in metas:
    meta=json.load(open(mp)); H,W=int(meta['height']),int(meta['width'])
    stem=mp.split('/')[-1].replace('_meta.json','')
    with rasterio.open(f'data/tcd/images/data/tcd/val/{stem}.tif') as s: img=np.transpose(s.read([1,2,3]),(1,2,0)).astype(np.uint8)
    x=((torch.from_numpy(img).permute(2,0,1).float()/255-mean)/std).unsqueeze(0).to(dev)
    with torch.no_grad(): prob=m(x).float().softmax(1)[0,1].cpu().numpy()
    pred=(prob>=0.5)
    # GT all-cat1 mask
    gt=np.zeros((H,W),np.uint8); nscored=0
    for cat,seg,bbox,area in B._parse_coco_annotations(meta):
        if cat!=1: continue
        if isinstance(seg,dict) and 'counts' in seg:
            r=seg
            if isinstance(r['counts'],list): r=mu.frPyObjects(r,r['size'][0],r['size'][1])
            gt|=mu.decode(r).astype(np.uint8)
        else:
            pl=B._seg_to_polygons(seg,H,W)
            if pl: gt|=B._rasterize(pl,H,W).astype(np.uint8); nscored+=1
    inter=(pred&(gt>0)).sum(); pf1=2*inter/(pred.sum()+gt.sum()+1e-9)
    n,_,st,_=cv2.connectedComponentsWithStats(pred.astype(np.uint8),8)
    nblob=sum(1 for k in range(1,n) if st[k,cv2.CC_STAT_AREA]>=256)
    f1s.append(pf1); ncc.append(nblob); ngt.append(nscored); covfrac.append(pred.mean())
print(f'holdout pixel-F1 vs all-cat1 GT: mean={np.mean(f1s):.3f}  (monitor val={round(ck.get("val_tree_F1",0),3)})')
print(f'pred canopy coverage frac: mean={np.mean(covfrac):.2f}')
print(f'pred blobs/tile (>=256px): mean={np.mean(ncc):.1f}  vs GT scored canopy/tile: mean={np.mean(ngt):.1f}')
