import sys, json, argparse, io; from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0,'phase30'); sys.path.insert(0,'.')
import benchmark as B
from canopy_aggregation_test import build_gt_anns, build_tree_dets, coco_eval_per_category, HOLDOUT_DIR, SCORE_THRESH
ap=argparse.ArgumentParser(); ap.add_argument('--dirs',required=True); a=ap.parse_args()
metas=sorted(HOLDOUT_DIR.glob('*_meta.json'))
images=[{'id':i,'file_name':m.name,'width':0,'height':0} for i,m in enumerate(metas)]
for d in a.dirs.split(','):
    gt=[]; dets=[]
    for i,mp in enumerate(metas):
        meta=json.load(open(mp)); H,W=int(meta['height']),int(meta['width']); stem=mp.name.replace('_meta.json','')
        preds=B._load_predictions(Path(d)/f'{stem}_canopyai.geojson',H,W,SCORE_THRESH)
        gt_to,_=build_gt_anns(meta,i,H,W,tree_only=True); gt+=gt_to
        dets+=build_tree_dets(preds,H,W,i,tree_cat=1)
    with redirect_stdout(io.StringIO()):
        ap50=coco_eval_per_category(images,gt,dets,[{'id':1,'name':'tree'}]).get(1,0.0)
    print(f'{d.split("/")[-1]:>30}  tree mAP50 = {ap50:.4f}  (n_dets={len(dets)})')
