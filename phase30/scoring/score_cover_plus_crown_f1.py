import os, sys, json, glob, time; os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_DATASETS_OFFLINE']='1'
sys.path.insert(0,'shadow_semseg')
import numpy as np, cv2, torch
from datasets import load_dataset
from config import Config; from model import build_model
dev='mps' if torch.backends.mps.is_available() else 'cpu'
def loadm(p):
    ck=torch.load(p,map_location=dev); cfg=Config(**{k:v for k,v in ck['cfg'].items() if k in Config.__dataclass_fields__})
    mm=build_model(cfg,verbose=False).to(dev).eval(); mm.load_state_dict(ck['model']); return mm,cfg
cover,cfg=loadm('/tmp/shadow_semseg/runs/semseg_v3_1024/best.pt')
canp,_=loadm('/tmp/shadow_semseg/runs/canopyreg_p22/best.pt')
mean=torch.tensor(cfg.imagenet_mean).view(3,1,1); std=torch.tensor(cfg.imagenet_std).view(3,1,1)
id2stem={}
for mp in glob.glob('data/tcd/images/data/tcd/val/*_meta.json'):
    mt=json.load(open(mp)); id2stem[mt['image_id']]=mp.split('/')[-1].replace('_meta.json','')
def crown_mask(stem,H,W,smin=0.1):
    f=f'benchmark_results_holdout/ablation_tcd_s0/{stem}_canopyai.geojson'; msk=np.zeros((H,W),np.uint8)
    if not stem or not os.path.exists(f): return msk
    for ft in json.load(open(f))['features']:
        if ft['properties'].get('deepforest_score',1.0)<smin: continue
        for ring in ft['geometry']['coordinates']:
            pts=np.array(ring).round().astype(np.int32)
            if len(pts)>=3: cv2.fillPoly(msk,[pts],1)
    return msk
def pred(model,x):
    with torch.no_grad(): return model(x).float().softmax(1)[0,1].cpu().numpy()>=0.5
ds=load_dataset('restor/tcd',split='test'); n=len(ds)
acc={k:[0,0,0] for k in ['cover','cover+crown','canopy_p22','canopyp22+crown']}
t0=time.time()
for i in range(n):
    ex=ds[i]; img=np.array(ex['image'].convert('RGB'),dtype=np.uint8); H,W=img.shape[:2]
    gt=(np.array(ex['annotation'].convert('L'))>0)
    x=((torch.from_numpy(img).permute(2,0,1).float()/255-mean)/std).unsqueeze(0).to(dev)
    cov=pred(cover,x); cp=pred(canp,x); cm=crown_mask(id2stem.get(ex['image_id']),H,W)>0
    def upd(k,p):
        acc[k][0]+=int((p&gt).sum()); acc[k][1]+=int((p&~gt).sum()); acc[k][2]+=int((~p&gt).sum())
    upd('cover',cov); upd('cover+crown',cov|cm); upd('canopy_p22',cp); upd('canopyp22+crown',cp|cm)
    if (i+1)%80==0 or (i+1)==n: print(f'  {i+1}/{n} {(time.time()-t0)/(i+1):.2f}s/tile',flush=True)
print('\n'+'='*64); print(f'{"set":>17} {"P":>7}{"R":>7}{"F1":>8}   [Restor 0.897/0.902]')
for k in ['cover','cover+crown','canopy_p22','canopyp22+crown']:
    tp,fp,fn=acc[k]; p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); f1=2*tp/(2*tp+fp+fn+1e-9)
    print(f'{k:>17} {p:>7.4f}{r:>7.4f}{f1:>8.4f}')
