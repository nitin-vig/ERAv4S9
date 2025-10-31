"""
Download ImageNet-1k from Hugging Face and convert to folder structure

This script downloads ImageNet-1k from Hugging Face datasets and 
organizes it into the expected folder structure for the training pipeline.

Usage:
    python download_imagenet1k_hf.py --data-root ./data
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("❌ Please install datasets library: pip install datasets")


def get_hf_cache_path():
    """Get Hugging Face datasets cache path from environment or default"""
    hf_datasets_cache = os.environ.get('HF_DATASETS_CACHE')
    if hf_datasets_cache:
        return hf_datasets_cache
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        return os.path.join(hf_home, 'datasets')
    # Default location
    return os.path.expanduser('~/.cache/huggingface/datasets')


def download_and_convert_imagenet1k(data_root, split='train', hf_cache_path=None):
    """
    Download ImageNet-1k from Hugging Face and save to folder structure
    
    Args:
        data_root: Root directory (e.g., './data')
        split: 'train' or 'validation'
        hf_cache_path: Optional path to HF datasets cache
    """
    if not HAS_DATASETS:
        print("❌ datasets library not installed")
        return False
    
    imagenet1k_path = os.path.join(data_root, "imagenet1k")
    split_dir = os.path.join(imagenet1k_path, split if split != 'validation' else 'val')
    os.makedirs(split_dir, exist_ok=True)
    
    # Check if already converted
    if os.path.exists(split_dir) and os.listdir(split_dir):
        class_folders = [d for d in os.listdir(split_dir) 
                        if os.path.isdir(os.path.join(split_dir, d))]
        if len(class_folders) > 0:
            print(f"✅ {split.capitalize()} split already exists at {split_dir}")
            print(f"   Found {len(class_folders)} class folders")
            return True
    
    # Set HF cache path if provided
    if hf_cache_path:
        os.environ['HF_DATASETS_CACHE'] = hf_cache_path
        print(f"📂 Using HF cache: {hf_cache_path}")
    
    print(f"📥 Loading ImageNet-1k {split} split from Hugging Face...")
    print("   (Using cached dataset if available, otherwise downloading)")
    
    try:
        # Load dataset from Hugging Face (will use cache if available)
        # Note: You'll need to accept the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            trust_remote_code=True
        )
        
        print(f"✅ Loaded {len(dataset)} images")
        print(f"📁 Saving images to {split_dir}...")
        
        # Create class folders and save images
        class_folders = {}
        for idx, item in enumerate(tqdm(dataset, desc=f"Saving {split}")):
            label = item['label']
            image = item['image']
            
            # Get class name (synset ID) from label
            # The label is already an index (0-999), but we need the synset name
            # For now, we'll use the label directly as folder name
            # You might want to map to actual synset IDs later
            class_name = f"n{label:08d}"  # Temporary naming
            
            # Or use the int2str method if available
            if hasattr(dataset.features['label'], 'int2str'):
                try:
                    class_name = dataset.features['label'].int2str(label)
                except:
                    pass
            
            class_dir = os.path.join(split_dir, class_name)
            if class_name not in class_folders:
                os.makedirs(class_dir, exist_ok=True)
                class_folders[class_name] = True
            
            # Save image
            img_filename = f"{idx:08d}.JPEG"
            img_path = os.path.join(class_dir, img_filename)
            image.save(img_path, "JPEG")
        
        print(f"✅ Saved {len(dataset)} images to {split_dir}")
        print(f"   Created {len(class_folders)} class folders")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        print("\n📝 Note: You may need to:")
        print("   1. Accept the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("   2. Login to Hugging Face: huggingface-cli login")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ImageNet-1k from Hugging Face and convert to folder structure'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help='Root directory for data (default: ./data)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Download only training split'
    )
    parser.add_argument(
        '--val-only',
        action='store_true',
        help='Download only validation split'
    )
    parser.add_argument(
        '--hf-cache',
        type=str,
        default=None,
        help='Path to Hugging Face datasets cache (uses HF_DATASETS_CACHE env var if not specified)'
    )
    parser.add_argument(
        '--use-existing-cache',
        action='store_true',
        help='Use existing HF cache location from environment variables'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Determine HF cache path
    hf_cache = args.hf_cache
    if args.use_existing_cache or hf_cache is None:
        hf_cache = get_hf_cache_path()
    
    print("="*80)
    print("ImageNet-1k Converter from Hugging Face Cache")
    print("="*80)
    print(f"Data root: {data_root}")
    print(f"Target path: {data_root / 'imagenet1k'}")
    print(f"HF cache: {hf_cache}")
    
    # Check if cache exists
    cache_exists = os.path.exists(hf_cache)
    if cache_exists:
        print(f"✅ HF cache found at: {hf_cache}")
    else:
        print(f"⚠️  HF cache not found at: {hf_cache}")
        print("   Will download if needed")
    
    print("\n📝 Notes:")
    print("   1. If dataset is already in HF cache, will use it (no re-download)")
    print("   2. If not cached, will download ~140GB+ of data")
    print("   3. Make sure you have accepted terms at:")
    print("      https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    print("   4. May need to login: huggingface-cli login")
    print("   5. Ensure sufficient disk space\n")
    
    success = True
    
    # Download train split
    if not args.val_only:
        print("\n" + "="*80)
        print("📥 Processing TRAIN split...")
        print("="*80)
        if not download_and_convert_imagenet1k(str(data_root), split='train', hf_cache_path=hf_cache):
            success = False
    
    # Download validation split
    if not args.train_only:
        print("\n" + "="*80)
        print("📥 Processing VALIDATION split...")
        print("="*80)
        if not download_and_convert_imagenet1k(str(data_root), split='validation', hf_cache_path=hf_cache):
            success = False
    
    if success:
        print("\n" + "="*80)
        print("✅ ImageNet-1k download complete!")
        print("="*80)
        imagenet1k_path = data_root / "imagenet1k"
        print(f"\n📁 Dataset location: {imagenet1k_path}")
        print(f"   Structure:")
        print(f"   {imagenet1k_path}/")
        print(f"   ├── train/")
        print(f"   │   ├── n01440764/")
        print(f"   │   └── ... (1000 class folders)")
        print(f"   └── val/")
        print(f"       ├── n01440764/")
        print(f"       └── ... (1000 class folders)")
        print(f"\n💡 You can now use:")
        print(f"   python train_imagenet_ec2.py --data-root {data_root}")
    else:
        print("\n❌ Download incomplete. Check errors above.")


if __name__ == '__main__':
    main()

