import os
import re
import win32com.client as win32


def slide_to_jpg(powerpoint, pptx_path, out_root):
    deck_name = sanitize_name(os.path.splitext(os.path.basename(pptx_path))[0])
    dest = os.path.join(out_root, deck_name)
    os.makedirs(dest, exist_ok=True)

    cur_pptx = None
    try:
        # Open the deck (ReadOnly, no window popping up)
        cur_pptx = powerpoint.Presentations.Open(
            pptx_path,
            ReadOnly=True,
            Untitled=False,
            WithWindow=False,
        )
        # Export each slide as JPG (1920x1080)
        cur_pptx.Export(dest, "JPG", 1920, 1080)
    finally:
        if cur_pptx is not None:
            cur_pptx.Close()


def sanitize_name(name: str) -> str:
    # Remove characters Windows doesn't like in folder names
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name).strip()


def main():
    # Base directory = folder this script lives in
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Folder with your .pptx/.ppt files (image_pipeline/slides)
    SLIDES_FOLDER = os.path.join(BASE_DIR, "slides")

    # Where you want the exported JPGs (image_pipeline/imgs)
    JPG_SAVE_FOLDER = os.path.join(BASE_DIR, "imgs")
    os.makedirs(JPG_SAVE_FOLDER, exist_ok=True)

    powerpoint = win32.Dispatch("PowerPoint.Application")
    powerpoint.Visible = True
    powerpoint.WindowState = 2  # minimized

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
