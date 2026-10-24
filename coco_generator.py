import pandas as pd
import json
import os
import hashlib
from sklearn.model_selection import train_test_split
import shutil
import glob
from PIL import Image

# === Function 1: Convert CSV to COCO JSON ===
def csv_to_coco(csv_path, output_json_path, datasets_root):
    """
    csv_path: CSV with columns [image_name, Xmin, Ymin, width, height, label, source_folder]
    datasets_root: root directory that contains all source_folder subdirs
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df = df.rename(columns={
        'Filename': 'image_name',
        'chip_position_x(px)': 'Xmin',
        'chip_position_y(px)': 'Ymin',
        'width(px)': 'width',
        'height(px)': 'height',
        'area(px)': 'area'
    })

    # Ensure source_folder exists in this CSV
    if 'source_folder' not in df.columns:
        raise ValueError("CSV must include a 'source_folder' column (absolute or datasets_root-relative).")

    # Manual label mapping
    label_mapping = {
        'Intermediate': 'Less_Thick_BN',
        'Thick': 'Thick_BN',
        'Thin': 'Thin_BN'
    }

    categories = [
        {"id": 1, "name": "Less_Thick_BN", "supercategory": "flakes"},
        {"id": 2, "name": "Thick_BN", "supercategory": "flakes"},
        {"id": 3, "name": "Thin_BN", "supercategory": "flakes"}
    ]
    category_to_id = {"Less_Thick_BN": 1, "Thick_BN": 2, "Thin_BN": 3}

    images, annotations = [], []
    image_id_map = {}  # key: uid, val: image_id
    annotation_id = 1

    for _, row in df.iterrows():
        orig_label = row['label']
        if orig_label not in label_mapping:
            # Skip unknown labels
            continue
        mapped_label = label_mapping[orig_label]

        image_name = str(row['image_name'])
        # Make source_folder absolute (row['source_folder'] is already relative to datasets_root)
        source_folder = str(row['source_folder']).strip()
        if not os.path.isabs(source_folder):
            source_folder = os.path.join(datasets_root, source_folder)
        source_folder = os.path.normpath(source_folder)  # normalize separators / resolve .. 
        image_path = os.path.join(source_folder, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️ Skipping {image_name} — not found at {image_path}")
            continue

        # Unique ID per (folder,file)
        uid = f"{source_folder}::{image_name}"

        # file_name stored in COCO should be a path we can recreate; keep it relative to datasets_root
        rel_path = os.path.relpath(image_path, start=datasets_root)

        # Read true dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"⚠️ Could not read {image_path}: {e}")
            continue

        if uid not in image_id_map:
            image_id = int(hashlib.sha256(uid.encode()).hexdigest(), 16) % 10**8
            image_id_map[uid] = image_id
            images.append({
                "id": image_id,
                "file_name": rel_path,  # keep relative path to maintain uniqueness
                "width": int(width),
                "height": int(height)
            })

        x, y, w, h = float(row['Xmin']), float(row['Ymin']), float(row['width']), float(row['height'])
        annotations.append({
            "id": annotation_id,
            "image_id": image_id_map[uid],
            "category_id": category_to_id[mapped_label],
            "bbox": [x, y, w, h],
            "segmentation": [],  # ok for detection-only
            "area": float(w * h),
            "iscrowd": 0
        })
        annotation_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"✅ COCO JSON saved to: {output_json_path}")


# === Function 2: Split CSV and generate COCO JSON files ===
def split_csv_and_generate_coco(csv_path, output_dir, datasets_root, seed=42):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        'Filename': 'image_name',
        'chip_position_x(px)': 'Xmin',
        'chip_position_y(px)': 'Ymin',
        'width(px)': 'width',
        'height(px)': 'height',
        'area(px)': 'area'
    })
    if 'source_folder' not in df.columns:
        raise ValueError("CSV must include 'source_folder'.")

    # Composite key to avoid filename collisions across folders
    df['uid'] = df.apply(lambda r: f"{r['source_folder']}::{r['image_name']}", axis=1)

    unique_uids = df['uid'].unique()
    train_uids, temp_uids = train_test_split(unique_uids, test_size=0.3, random_state=seed)
    valid_uids, test_uids = train_test_split(temp_uids, test_size=0.5, random_state=seed)

    train_df = df[df['uid'].isin(train_uids)].drop(columns=['uid'])
    valid_df = df[df['uid'].isin(valid_uids)].drop(columns=['uid'])
    test_df  = df[df['uid'].isin(test_uids)].drop(columns=['uid'])

    os.makedirs(output_dir, exist_ok=True)
    train_csv = os.path.join(output_dir, 'train.csv')
    valid_csv = os.path.join(output_dir, 'valid.csv')
    test_csv  = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print("✅ CSV splits saved.")

    csv_to_coco(train_csv, os.path.join(output_dir, 'train_annotations.coco.json'), datasets_root)
    csv_to_coco(valid_csv, os.path.join(output_dir, 'valid_annotations.coco.json'), datasets_root)
    csv_to_coco(test_csv,  os.path.join(output_dir, 'test_annotations.coco.json'),  datasets_root)


# === Function 3: Move image files into split folders (exact paths) ===
def move_images_by_csv(csv_dir, datasets_root):
    """
    Copies the exact files referenced by each split CSV into {csv_dir}/{split}_images,
    preserving subfolder structure under datasets_root.
    """
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        csv_path = os.path.join(csv_dir, f"{subset}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'source_folder' not in df.columns:
            raise ValueError(f"{csv_path} is missing 'source_folder'.")

        dest_folder = os.path.join(csv_dir, f"{subset}_images")
        os.makedirs(dest_folder, exist_ok=True)

        # Use per-row exact path; dedupe by rel path
        seen = set()
        for _, r in df.iterrows():
            src = os.path.join(r['source_folder'], r['image_name'])
            if not os.path.isabs(src):
                src = os.path.join(datasets_root, src)
            if not os.path.exists(src):
                print(f"⚠️ Missing image: {src}")
                continue
            rel = os.path.relpath(src, start=datasets_root)
            if rel in seen:
                continue
            seen.add(rel)
            dst = os.path.join(dest_folder, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        print(f"✅ {subset.capitalize()} images copied to: {dest_folder}")


# === Function 4: Organize final data folder ===
def organize_data_folders(output_dir, datasets_root):
    """
    Create Data/{train,valid,test}, copy COCOs and images (preserving subfolders).
    """
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join('Data', split)
        os.makedirs(split_dir, exist_ok=True)

        # Copy COCO annotation file
        coco_src = os.path.join(output_dir, f"{split}_annotations.coco.json")
        coco_dst = os.path.join(split_dir, "_annotations.coco.json")
        if os.path.exists(coco_src):
            shutil.copyfile(coco_src, coco_dst)
            print(f"✅ Copied COCO annotation to: {coco_dst}")
        else:
            print(f"⚠️ Missing annotation file for {split}: {coco_src}")

        # Copy images exactly as referenced in CSV
        csv_path = os.path.join(output_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing CSV for {split}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if 'source_folder' not in df.columns:
            raise ValueError(f"{csv_path} is missing 'source_folder'.")

        seen = set()
        for _, r in df.iterrows():
            src = os.path.join(r['source_folder'], r['image_name'])
            if not os.path.isabs(src):
                src = os.path.join(datasets_root, src)
            if not os.path.exists(src):
                print(f"⚠️ Missing image: {src}")
                continue
            rel = os.path.relpath(src, start=datasets_root)
            if rel in seen:
                continue
            seen.add(rel)
            dst = os.path.join(split_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)


# === Function 5: Cleanup ===
def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"🗑️  Deleted temporary folder: {folder_path}")
    else:
        print(f"⚠️ Folder not found for deletion: {folder_path}")


def merge_multiple_csvs_and_generate_coco(datasets_dir, output_dir):
    """
    Expect structure:
      datasets/
        setA.csv
        setA/  (images for setA.csv)
        setB.csv
        setB/  (images for setB.csv)
        ...
    """
    merged_df = pd.DataFrame()

    for csv_file in glob.glob(os.path.join(datasets_dir, "*.csv")):
        df = pd.read_csv(csv_file)
        base = os.path.splitext(os.path.basename(csv_file))[0]
        image_dir = os.path.join(datasets_dir, base)
        if not os.path.isdir(image_dir):
            print(f"⚠️ Skipping {csv_file} — no matching image folder: {image_dir}")
            continue
        # store source_folder as datasets_root-relative
        df['source_folder'] = os.path.relpath(image_dir, start=datasets_dir)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_csv = os.path.join(output_dir, "merged.csv")
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(merged_csv, index=False)
    print(f"✅ Merged CSV saved to {merged_csv}")
    return merged_csv


if __name__ == "__main__":
    datasets_root = "datasets"
    output_dir = "COCO_exports"

    # 1) Merge datasets/* pairs into one CSV (with source_folder retained)
    merged_csv = merge_multiple_csvs_and_generate_coco(datasets_root, output_dir)

    # 2) Split + generate COCO JSONs (uses datasets_root for true image dims & rel paths)
    split_csv_and_generate_coco(merged_csv, output_dir, datasets_root)

    # 3) Copy split images to split folders for quick inspection (optional)
    move_images_by_csv(output_dir, datasets_root)

    # 4) Create final Data/{train,valid,test} bundles
    organize_data_folders(output_dir, datasets_root)

    # 5) (Optional) 
    delete_folder(output_dir)
