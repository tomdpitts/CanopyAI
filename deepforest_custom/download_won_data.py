import modal
import os
import shutil
import tarfile


def download_data():
    """Download WON003 data from Modal volume for local testing."""
    print("🚀 Connecting to Modal volume...")
    vol = modal.Volume.from_name("canopyai-deepforest-data")

    local_dir = "/tmp/won_data"
    os.makedirs(local_dir, exist_ok=True)

    files_to_download = ["won_train.csv", "won_val.csv", "won_images.tar.gz"]

    for filename in files_to_download:
        print(f"📥 Downloading {filename}...")
        try:
            # Modal volume read returns bytes
            data = b""
            for chunk in vol.read_file(filename):
                data += chunk

            with open(os.path.join(local_dir, filename), "wb") as f:
                f.write(data)
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return

    print("\n📦 Extracting images...")
    try:
        with tarfile.open(os.path.join(local_dir, "won_images.tar.gz"), "r:gz") as tar:
            tar.extractall(path=local_dir)
        print("✅ Extracted images")
    except Exception as e:
        print(f"❌ Error extracting images: {e}")
        return

    print(f"\n✨ Data ready in {local_dir}")
    print(
        f"   CSVs: {os.path.join(local_dir, 'won_train.csv')}, {os.path.join(local_dir, 'won_val.csv')}"
    )
    # Check extraction path
    extracted_dirs = [
        d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))
    ]
    print(f"   Extracted directories: {extracted_dirs}")


if __name__ == "__main__":
    download_data()
