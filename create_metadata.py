import os
import pandas as pd

# Define paths
base_dir = "C:/Users/Akoba/Desktop/START up/COMPUTER-AIDED-VISION-FOR-PLANT-DISEASE-DETECTION"
data_dir = os.path.join(base_dir, "Data/PlantVillage")
output_csv = os.path.join(base_dir, "Data/PlantVillage/metadata.csv")

# Make sure the output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Supported image extensions (case-insensitive)
image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")

# Collect image paths and labels with logging
image_paths, labels = [], []
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        print(f"Processing folder: {class_name}")
        img_count = 0
        for img in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img)
            # Check if it's a file and has a valid image extension
            if os.path.isfile(img_path) and img.lower().endswith(image_extensions):
                # Use absolute path for now to ensure capture, can switch to relative later
                # rel_path = os.path.relpath(img_path, base_dir)
                image_paths.append(img_path)  # Absolute path for debugging
                labels.append(class_name)
                img_count += 1
            else:
                print(f"Skipped: {img_path} (not an image or invalid extension)")
        print(f"Found {img_count} images in {class_name}")
    else:
        print(f"Skipped: {class_dir} (not a directory)")

# Save to CSV
df = pd.DataFrame({"image_path": image_paths, "label": labels})
df.to_csv(output_csv, index=False)
print(f"✅ metadata.csv created with {len(df)} images across {df['label'].nunique()} classes.")

# Additional diagnostics
print("\nClass distribution:")
print(df["label"].value_counts())