import os
import site
import glob

def patch_imagecorruptions():
    # 1. Start by finding where the `imagecorruptions` package is installed
    # It checks all site-packages directories (including virtual environments)
    packages_dirs = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        packages_dirs.append(site.getusersitepackages())
        
    # We also check the current environment sys.path natively just in case
    import sys
    packages_dirs.extend(sys.path)

    target_file = None
    for d in packages_dirs:
        possible_path = os.path.join(d, 'imagecorruptions', 'corruptions.py')
        if os.path.exists(possible_path):
            target_file = possible_path
            break
            
    if not target_file:
        print("Could not find 'imagecorruptions' package installed. Are you in the right environment?")
        return
        
    print(f"Found corruptions.py at: {target_file}")
    
    # 2. Read the script
    with open(target_file, 'r') as f:
        file_data = f.read()
        
    # 3. Apply the backward/forward compatibility fixes
    # In skimage 0.20+, 'multichannel=True' became 'channel_axis=-1'
    if "multichannel=True" in file_data or "multichannel=False" in file_data:
        print("Legacy 'multichannel' arguments found! Patching for Python 3.13/skimage 0.20+ compatibility...")
        
        # Replace True multichannel flag (RGB Images)
        file_data = file_data.replace("multichannel=True", "channel_axis=-1")
        # Replace False multichannel flag (Grayscale operations)
        file_data = file_data.replace("multichannel=False", "channel_axis=None")
        
        # FIX 2: Numpy 2.0+ deprecations
        # imagecorruptions uses np.float_, np.int_, np.bool_ which blow up under modern numpy runtimes
        if "np.float_" in file_data:
            print("Legacy 'np.float_' Numpy types found! Patching to 'np.float64'...")
            file_data = file_data.replace("np.float_", "np.float64")
            
        if "np.int_" in file_data:
            file_data = file_data.replace("np.int_", "np.int64")
            
        if "np.bool_" in file_data:
            file_data = file_data.replace("np.bool_", "bool")
            
        if "np.math.factorial" in file_data or "np.math." in file_data:
             # Just in case they used np.math which was stripped from top-level namespace
             file_data = file_data.replace("np.math.", "math.")
             if "import math" not in file_data:
                 file_data = "import math\n" + file_data

        if "from scipy.ndimage.interpolation import map_coordinates" in file_data:
            print("Legacy 'scipy.ndimage.interpolation' found! Patching to 'scipy.ndimage'...")
            file_data = file_data.replace("from scipy.ndimage.interpolation import map_coordinates", "from scipy.ndimage import map_coordinates")

        # 4. Write it back exactly in the virtual environment
        with open(target_file, 'w') as f:
            f.write(file_data)
            
        print("Successfully patched 'imagecorruptions'! You can now run the pipeline on Python 3.13.")
    else:
        print("The file doesn't contain 'multichannel=' arguments. It might already be patched.")

if __name__ == "__main__":
    print("--- Starting Compatibility Patcher for ImageCorruptions ---")
    patch_imagecorruptions()