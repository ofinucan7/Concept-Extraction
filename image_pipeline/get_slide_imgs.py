import os
import win32com.client as win32
from win32com.client import constants
import re

def slide_to_jpg(powerpoint, pptx_path, out_root):
    deck_name = sanitize_name(os.path.splitext(os.path.basename(pptx_path))[0])
    dest = os.path.join(out_root, deck_name)
    os.makedirs(dest, exist_ok=True)

    # try to save a JPG of the cur powerpoint
    cur_pptx = None
    try:
        cur_pptx = powerpoint.Presentations.Open(pptx_path, ReadOnly=True, Untitled=False, WithWindow=False)
        cur_pptx.Export(dest, "JPG", 1920, 1080)
    finally:
        if cur_pptx is not None:
            cur_pptx.Close()

def sanitize_name(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name).strip()

def main():
    # slides_folder --> folder with all your pptx files
    # jpg_save_folder --> where you want all the 
    SLIDES_FOLDER = r"C:\Users\ofinu\Documents\concept-extraction\image-grabber\slides"
    JPG_SAVE_FOLDER = r"C:\Users\ofinu\Documents\concept-extraction\image-grabber\imgs"
    os.makedirs(JPG_SAVE_FOLDER, exist_ok=True)

    powerpoint = win32.gencache.EnsureDispatch("PowerPoint.Application")
    powerpoint.Visible = True              # don't hide; some builds forbid it
    powerpoint.WindowState = constants.ppWindowMinimized  # or use 2 if you prefer: powerpoint.WindowState = 2
    num_outputted = 0

    try:
        for name in sorted(os.listdir(SLIDES_FOLDER)):
            if name.lower().endswith((".pptx", ".ppt")):
                src = os.path.join(SLIDES_FOLDER, name)
                slide_to_jpg(powerpoint, src, JPG_SAVE_FOLDER)
                num_outputted += 1
                print(f"outputted: {num_outputted} files")
    finally:
        powerpoint.Quit()

if __name__ == "__main__":
    main()
