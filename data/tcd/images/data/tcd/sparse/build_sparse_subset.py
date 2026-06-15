"""Materialise the 'sparse' TCD test subset: 180 tiles, biome NOT in {1,2,3,7,-1}
(no tropical biomes, no Unknown), NO canopy restriction, any tree count.
Symlinks {stem}.tif + {stem}_meta.json into a new folder (original stems kept),
writes a manifest + provenance JSON."""
import json, glob, os, datetime
from pycocotools import mask as M

REPO = "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
VAL  = os.path.join(REPO, "data/tcd/images/data/tcd/val")
DST  = os.path.join(REPO, "data/tcd/images/data/tcd/sparse")
MANIFEST = os.path.join(REPO, "sparse_tiles.txt")
EXCL = {1, 2, 3, 7, -1}          # tropical biomes + Unknown (NO canopy restriction)
BIOME = {1:"TropMoistForest",2:"TropDryForest",3:"TropConifForest",4:"TempBroadleaf",
         5:"TempConifer",6:"Boreal",7:"TropSavanna",8:"TempGrass",9:"FloodedGrass",
         10:"MontaneGrass",11:"Tundra",12:"Mediterranean",13:"Desert/Xeric",14:"Mangrove",-1:"Unknown"}

def rle(s,H,W):
    if isinstance(s,list): return M.merge(M.frPyObjects(s,H,W))
    if isinstance(s,dict): return M.frPyObjects(s,H,W) if isinstance(s.get("counts"),list) else s
def cf(anns,W,H):
    rs=[]
    for a in anns:
        if a.get("category_id")!=1 or not a.get("segmentation"): continue
        try:
            r=rle(a["segmentation"],H,W)
            if r is not None: rs.append(r)
        except: pass
    return float(M.area(M.merge(rs)))/(W*H) if rs else 0.0

sel=[]
for f in sorted(glob.glob(os.path.join(VAL,"*_meta.json"))):
    d=json.load(open(f))
    if d.get("biome") in EXCL: continue
    raw=d["coco_annotations"]; anns=json.loads(raw) if isinstance(raw,str) else raw
    c=cf(anns,d.get("width",2048),d.get("height",2048))   # recorded for provenance; not filtered
    nt=sum(1 for a in anns if a.get("category_id")==2)
    sel.append({"stem":os.path.basename(f).replace("_meta.json",""),"trees":nt,
                "canopy_frac":round(c,4),"biome":BIOME.get(d.get("biome"),"?"),"biome_code":d.get("biome")})
sel.sort(key=lambda r:(r["canopy_frac"],-r["trees"]))
print(f"selected {len(sel)} tiles")

os.makedirs(DST, exist_ok=True)
n_link=0
for r in sel:
    for ext in (".tif","_meta.json"):
        src=os.path.join(VAL, r["stem"]+ext); dst=os.path.join(DST, r["stem"]+ext)
        if os.path.lexists(dst): os.remove(dst)
        if os.path.exists(src): os.symlink(src, dst); n_link+=1
        else: print(f"  ! missing {src}")
with open(MANIFEST,"w") as fh:
    fh.write("\n".join(r["stem"] for r in sel)+"\n")

prov={
 "subset":"sparse","n_tiles":len(sel),
 "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
 "source_dir":"data/tcd/images/data/tcd/val",
 "criteria":{"canopy_frac":"no restriction (all densities; recorded per-tile for reference)",
             "biome_excluded":["TropMoistForest(1)","TropDryForest(2)","TropConifForest(3)",
                               "TropSavanna(7)","Unknown(-1)"],
             "tree_floor":"none (any tree count, incl. 0)"},
 "stems_kept_as":"tcd_val_tile_N (symlinks; preserves reuse of existing predictions via --tiles-file)",
 "tiles":sel,
}
json.dump(prov, open(os.path.join(DST,"_SUBSET_PROVENANCE.json"),"w"), indent=2)
from collections import Counter
print(f"symlinked {n_link} files into {DST}")
print("biome mix:", dict(Counter(r['biome'] for r in sel)))
print(f"tree floor: 0-tree tiles = {sum(1 for r in sel if r['trees']==0)}, >=1 = {sum(1 for r in sel if r['trees']>=1)}")
print(f"manifest: {MANIFEST}")
