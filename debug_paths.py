import modal
import os

app = modal.App("debug-paths")
volume = modal.Volume.from_name("canopyai-data")


@app.function(volumes={"/data": volume})
def list_paths():
    print("Listing /data:")
    print(os.listdir("/data"))

    if os.path.exists("/data/won"):
        print("\nListing /data/won:")
        print(os.listdir("/data/won"))

        if os.path.exists("/data/won/raw"):
            print("\nListing /data/won/raw (first 5):")
            print(os.listdir("/data/won/raw")[:5])
    else:
        print("\n/data/won does not exist")


@app.local_entrypoint()
def main():
    list_paths.remote()
