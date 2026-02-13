import pandas as pd
import os


def prepare_local_data():
    base_dir = "/tmp/won_data"

    for split in ["train", "val"]:
        csv_path = f"{base_dir}/won_{split}.csv"
        df = pd.read_csv(csv_path)

        # Rewrite paths to match local extraction
        # Original: /data/images/data/won/raw/...
        # Local: /tmp/won_data/data/won/raw/...
        df["image_path"] = df["image_path"].str.replace("/data/images", base_dir)

        # Verify images exist
        print(f"Checking {split} images...")
        missing = 0
        for path in df["image_path"].unique():
            if not os.path.exists(path):
                print(f"❌ Missing: {path}")
                missing += 1

        if missing == 0:
            print(f"✅ All {len(df['image_path'].unique())} images found for {split}!")
        else:
            print(f"⚠️  {missing} images missing for {split}")

        # Save fixed CSV
        output_path = f"{base_dir}/local_won_{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved fixed CSV to {output_path}")

        # Create tiny version for quick testing
        tiny_df = df.head(50)
        tiny_path = f"{base_dir}/local_won_{split}_tiny.csv"
        tiny_df.to_csv(tiny_path, index=False)
        print(f"💾 Saved tiny CSV to {tiny_path}")


if __name__ == "__main__":
    prepare_local_data()
