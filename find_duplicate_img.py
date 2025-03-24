import os
import hashlib
from PIL import Image

def get_image_hash(image_path):
    """Compute a hash for an image file."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((64, 64))  # Convert to grayscale and resize
        return hashlib.md5(img.tobytes()).hexdigest()  # Compute hash

def find_duplicate_images(folder_path):
    """Find duplicate images without deleting them."""
    image_hashes = {}
    duplicates = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                img_hash = get_image_hash(file_path)

                if img_hash in image_hashes:
                    duplicates.append((file_path, image_hashes[img_hash]))  # Store duplicates
                else:
                    image_hashes[img_hash] = file_path  # Store first occurrence

    if duplicates:
        print("\n🔍 Found duplicate images:")
        for dup, original in duplicates:
            print(f"❌ Duplicate: {dup}  🆚  ✅ Original: {original}")
    else:
        print("✅ No duplicates found.")

    print(f"\n🎯 Total duplicates found: {len(duplicates)}")

def remove_duplicate_images(folder_path):
    """Find and remove duplicate images, keeping only one copy."""
    image_hashes = {}
    duplicates = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                img_hash = get_image_hash(file_path)

                if img_hash in image_hashes:
                    duplicates.append(file_path)  # Mark for deletion
                else:
                    image_hashes[img_hash] = file_path
    
    # Remove duplicates (corrected)
    for dup in duplicates:
        full_path = os.path.join(folder_path, dup)  # Ensure full path
        if os.path.exists(full_path):  # Check if the file exists
            os.remove(full_path)
            print(f"✅ Removed duplicate: {full_path}")
        else:
            print(f"⚠️ File not found: {full_path}")

    print(f"✅ Process completed! {len(duplicates)} duplicates removed.")

def find_and_create_new_labels(folder_path, label_file):
    """Find duplicate images and create a new label file without affecting the original."""
    image_hashes = {}
    duplicates = set()  # Use a set to store only unique duplicate filenames

    # Step 1: Detect Duplicates
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                img_hash = get_image_hash(file_path)

                if img_hash in image_hashes:
                    duplicates.add(file)  # Store only exact filenames
                else:
                    image_hashes[img_hash] = file  # Store first occurrence

    # Step 2: Read original label.txt and create a new filtered label file
    new_label_file = os.path.join(os.path.dirname(label_file), "new_label.txt")

    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()

        # **Exact match check for duplicate filenames**
        updated_lines = [line for line in lines if line.split()[0] not in duplicates]

        # Step 3: Write to new label file
        with open(new_label_file, "w") as f:
            f.writelines(updated_lines)

        print(f"✅ Created new label file: {new_label_file}")
        print(f"🚀 {len(lines) - len(updated_lines)} duplicate records removed.")
    else:
        print("⚠️ label.txt not found!")

    # Step 4: Print found duplicates
    if duplicates:
        print("\n🔍 Found duplicate images:")
        for dup in sorted(duplicates):
            print(f"❌ Duplicate: {dup}")
    else:
        print("✅ No duplicates found.")
        
# Example usage
folder_path = "/home/sebastian/Desktop/NCCU_Smoke/final_smoke_datasets"
label_file = "/home/sebastian/Desktop/NCCU_Smoke/final_smoke_datasets_label/labels.txt"  # Change to your label file path
# find_and_create_new_labels(folder_path, label_file)
# find_duplicate_images(folder_path)
remove_duplicate_images(folder_path)
