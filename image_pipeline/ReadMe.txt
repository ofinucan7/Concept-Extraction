Note: For all of this to work, you will need an OpenAI API key with credits in a .env
OPENAI_API_KEY = --- your api key here ---

Scripts:
get_slide_imgs.py
    Takes screenshots of every slide in a powerpoint within "slides" folder, uses txt in 'labels' folder
    Outputs screenshots to imgs folder with subdirectory of that slideshow
    NOTE: you must have powerpoint installed to get this script to work
annotate_imgs.py
    Goes through and annotates slide screenshots in batches
    Takes in the imgs (from subfolders) from the '/imgs' root and makes sentences describing what is happening on those slide batches
    Outputs jsons of the annotations in the 'annotations' folder
get_concepts.py
    Outputs list of concepts from the provided data
    Takes in txts from prompt_helper_data, annotations from 'annotations' folder
    Outputs list of concepts (txt) to the 'concepts' folder
stats.py
    Calculates precision, recall, and f1 scores of data
    Takes in txts from 'stats' folder
    Prints outputs in terminal