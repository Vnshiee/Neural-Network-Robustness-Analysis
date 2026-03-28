import os
import glob
import shutil

def verify_and_structure_imagenet100(base_dir='./data'):
    """
    Since the professor strictly requires the official ImageNet-100 from Kaggle/HuggingFace,
    this script acts as a verifier and folder reorganizer.
    
    You must first manually download the Kaggle zip:
       kaggle datasets download -d ambityga/imagenet100
       unzip imagenet100.zip -d ./data/imagenet100_raw
    """
    target_dir = os.path.join(base_dir, 'imagenet100')
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    # Check if the target is already organized
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 10:
        print("✅ ImageNet-100 is already correctly structured at ./data/imagenet100 !!")
        return

    # Look for the unzipped Kaggle folder
    raw_candidates = [
        os.path.join(base_dir, 'imagenet100_raw'),
        './imagenet100_raw'
    ]
    
    raw_dir = None
    for c in raw_candidates:
        if os.path.exists(c):
            raw_dir = c
            break
            
    if not raw_dir:
        print("❌ Could not find the Kaggle raw files.")
        print("Please follow the Kaggle CLI download instructions first!")
        return
        
    print(f"Found raw unzipped data at: {raw_dir}")
    print("Molding into PyTorch ImageFolder layout...")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Depending on how the zip extracts, it usually has a 'train.X1', 'val.X' folders inside
    for folder in os.listdir(raw_dir):
        full_path = os.path.join(raw_dir, folder)
        if os.path.isdir(full_path):
            if 'train' in folder.lower():
                shutil.move(full_path, train_dir)
            elif 'val' in folder.lower():
                shutil.move(full_path, val_dir)

    print("====================================")
    print("✅ ImageNet-100 structured perfectly!")
    print(f"Location: {os.path.abspath(target_dir)}")
    print("====================================")

if __name__ == "__main__":
    verify_and_structure_imagenet100()