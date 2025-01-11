import os
import json
import shutil
from sklearn.model_selection import train_test_split

final_db_path = "Final Database"
output_images_path = "Final Database/combined/images"
output_annotations_path = "Final Database/combined/annotations"

os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_annotations_path, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_images_path, split), exist_ok=True)

# Load and combine annotations
all_annotations = []
all_images = []
image_id_offset = 0
annotation_id_offset = 0

for db in ["Database 1", "Database 2"]:
    db_annotations_path = os.path.join(final_db_path, db, "annotations", "instances_default.json")
    with open(db_annotations_path, "r") as f:
        data = json.load(f)
    
    # Update image and annotation IDs to avoid conflicts
    for img in data["images"]:
        img["id"] += image_id_offset
        all_images.append(img)
    for ann in data["annotations"]:
        ann["image_id"] += image_id_offset
        ann["id"] += annotation_id_offset
        all_annotations.append(ann)
    
    image_id_offset += len(data["images"])
    annotation_id_offset += len(data["annotations"])

# Combine into a single COCO structure
coco_combined = {
    "images": all_images,
    "annotations": all_annotations,
    "categories": data["categories"]
}

# Split into train, val, test
image_ids = [img["id"] for img in all_images]
train_ids, temp_ids = train_test_split(image_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

splits = {"train": train_ids, "val": val_ids, "test": test_ids}
for split, ids in splits.items():
    split_images = [img for img in all_images if img["id"] in ids]
    split_annotations = [ann for ann in all_annotations if ann["image_id"] in ids]
    
    # Save the split annotations
    coco_split = {
        "images": split_images,
        "annotations": split_annotations,
        "categories": coco_combined["categories"]
    }
    with open(os.path.join(output_annotations_path, f"instances_{split}.json"), "w") as f:
        json.dump(coco_split, f)
    
    # Copy the corresponding images
    for img in split_images:
        # Determine the source database based on image file name
        src_img_path = None
        for db in ["Database 1", "Database 2"]:
            db_img_path = os.path.join(final_db_path, db, "images", img["file_name"])
            if os.path.exists(db_img_path):
                src_img_path = db_img_path
                break
        if src_img_path:
            dst_img_path = os.path.join(output_images_path, split, img["file_name"])
            shutil.copyfile(src_img_path, dst_img_path)

print("Datasets successfully split and organized!")
