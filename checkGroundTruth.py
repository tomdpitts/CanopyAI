# import json
# from huggingface_hub import hf_hub_url, list_files_info

# files = list_files_info("restor/tcd")
# for f in files[:10]:
#     if f.rfilename.endswith(".tif"):
#         print(f.rfilename)

import pprint


from datasets import load_dataset
ds = load_dataset("restor/tcd", split="train")
example = ds[0]
print(example.keys())

print(example["segments"][:300])