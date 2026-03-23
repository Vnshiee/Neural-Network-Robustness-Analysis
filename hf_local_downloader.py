import os
import shutil
from tqdm import tqdm
try:
    from datasets import load_dataset
except ImportError:
    print("Please install the datasets library first: pip install datasets")
    exit(1)

def build_hf_dataset():
    print("1. Downloading abstract ImageNet-100 dataset from Hugging Face...")
    # This will download the dataset and cache it locally
    ds = load_dataset("clane9/imagenet-100")
    
    # We will build the PyTorch ImageFolder structure locally before zipping
    base_dir = "./data/imagenet100"
    
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    print(f"2. Extracting raw JPGs into {base_dir} ...")
    
    # Hugging Face provides 'train' and 'validation' splits
    splits = {'train': 'train', 'validation': 'val'}
    
    for hf_split, folder_split in splits.items():
        split_dir = os.path.join(base_dir, folder_split)
        os.makedirs(split_dir, exist_ok=True)
        
        dataset_split = ds[hf_split]
        print(f"Exporting {folder_split} data ({len(dataset_split)} images)...")
        
        for i, item in enumerate(tqdm(dataset_split)):
            image = item['image']
            label = item['label']
            
            # Map labels to structural folders
            class_dir = os.path.join(split_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            
            # Save the image as JPEG
            img_path = os.path.join(class_dir, f"img_{i}.jpg")
            
            # Convert grayscale / RGBA to strict RGB to prevent ImageNet loading crashes
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            image.save(img_path, 'JPEG')
            
    print("3. Compressing the completed ImageNet-100 structure into a lightweight ZIP...")
    shutil.make_archive('imagenet100_ready', 'zip', base_dir)
    
    print("====================================")
    print("✅ SUCCESS! 'imagenet100_ready.zip' created!")
    print("====================================")

if __name__ == "__main__":
    build_hf_dataset()