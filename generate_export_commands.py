import os
import sys


LDRAW_LIBRARY_PATH = "C:\Program Files\LDraw (64 Bit)\library"
OUTPUT_DAE_DIR = "Input"
LEOCAD_EXECUTABLE = "leocad.exe" # Use "leocad.exe" on Windows

def find_ldraw_parts(lib_path):
    """Finds all .dat files in the 'parts' and 'p' subdirectories."""
    part_files = []
    
    # Check common locations within the LDraw path
    subdirs = ['parts', 'p']
    
    for subdir in subdirs:
        search_path = os.path.join(lib_path, subdir)
        if os.path.isdir(search_path):
            print(f"[INFO] Searching: {search_path}")
            for root, _, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(('.dat', '.ldr', '.l3p')):
                        # Store the full path to the part file
                        part_files.append(os.path.join(root, file))
    return part_files

def generate_batch_script(part_files, ldraw_path, out_dir, leocad_exe):
    """Generates a shell script to run LeoCAD commands."""
    
    # Use '.sh' for Unix/Linux or '.bat' for Windows
    if sys.platform.startswith('win'):
        script_name = "batch_export_dae.bat"
        command_template = '"{exe}" -l "{lib}" "{part}" -dae "{out}"\n'
    else:
        script_name = "batch_export_dae.sh"
        command_template = '"{exe}" -l "{lib}" "{part}" -dae "{out}"\n'
        
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[INFO] Generating {script_name} with {len(part_files)} commands...")
    
    with open(script_name, 'w') as f:
        # Optional header for batch/shell script
        if script_name.endswith('.sh'):
            f.write("#!/bin/bash\n\n")
            
        for part_path in part_files:
            # Create a unique, clean output filename (e.g., 3001.dae)
            part_filename = os.path.basename(part_path)
            base_name = os.path.splitext(part_filename)[0]
            output_dae_path = os.path.join(out_dir, f"{base_name}.dae")
            
            command = command_template.format(
                exe=leocad_exe,
                lib=ldraw_path,
                part=part_path,
                out=output_dae_path
            )
            f.write(command)
            
    print(f"[SUCCESS] Export script saved as {script_name}. Run this script to start the conversion.")


if __name__ == "__main__":
    if LDRAW_LIBRARY_PATH == "/path/to/your/LDraw/library/folder":
        print("[ERROR] Please set the LDRAW_LIBRARY_PATH variable in the script first.")
        sys.exit(1)
        
    all_parts = find_ldraw_parts(LDRAW_LIBRARY_PATH)
    if not all_parts:
        print(f"[ERROR] No LDraw parts found in: {LDRAW_LIBRARY_PATH}")
        sys.exit(1)
        
    generate_batch_script(all_parts, LDRAW_LIBRARY_PATH, OUTPUT_DAE_DIR, LEOCAD_EXECUTABLE)