# CanopyAI
Repo for UTS Tree Canopy Measuring AI tools

1. Set up Python virtual env -> Must be Python 3.12!
2. pip install requirements.txt
3. Ready to go: 
    a. python prepare_data.py   # (defaults to 20 images)
    b. python infer.py          # (defaults to Detectree2 model baseline)
    c. python train.py --preset tiny --weights baseline --already_downloaded

4. Once training has been done, use new weights with:
    d. python infer.py --weights finetuned

Note: prepare_data.py will download a number of images with the arid_rangeland filter. Use e.g. "--max_images 65" to adjust max number of images

