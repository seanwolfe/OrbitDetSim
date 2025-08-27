import os
import shutil

def extract_meta_files(src_folder, dst_folder, keyword="meta"):
    # Make sure destination exists
    os.makedirs(dst_folder, exist_ok=True)

    # Walk through all subfolders
    for root, _, files in os.walk(src_folder):
        for file in files:
            if keyword in file:  # case-insensitive match
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_folder, file)

                # If duplicate filename exists, rename with counter
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(os.path.join(dst_folder, f"{base}_{counter}{ext}")):
                        counter += 1
                    dst_path = os.path.join(dst_folder, f"{base}_{counter}{ext}")

                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")

if __name__ == "__main__":
    src_folder = "bh_plus_range"   # <-- change this
    dst_folder = "meta_files"  # <-- change this
    extract_meta_files(src_folder, dst_folder)
    print("Done!")
