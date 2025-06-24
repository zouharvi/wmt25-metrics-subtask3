# %%
import csv
import tqdm
import collections
from typing import Dict, List, Any

LANGPAIRS = {"en-cs", "en-is", "en-ja", "en-ru", "en-uk", "en-zh"}

# load data
with open("data/dev-wmt24-jun19.tsv", "r") as f:
    data = list(csv.DictReader(f, delimiter='\t'))

# { x["source_lang"] for x in data} | { x["target_lang"] for x in data}
lang2name = {
    'cs': "Czech",
    'de': "German",
    'en': "English",
    'es': "Spanish",
    'hi': "Hindi",
    'is': "Icelandic",
    'ja': "Japanese",
    'ru': "Russian",
    'uk': "Ukrainian",
    'zh': "Chinese",
}


CACHE = collections.defaultdict(lambda: collections.defaultdict(dict))

# %%

def translate_baseline0(line: Dict):
    """
    baseline0: Do no post-editing.
    """
    return line["hypothesis_segment"]

OUTPUT_BASELINE0 = []
for line_i, line in enumerate(tqdm.tqdm(data)):
    OUTPUT_BASELINE0.append(translate_baseline0(line))

# %%

from openai import OpenAI
from typing import Dict
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_baseline1(line: Dict):
    """
    baseline1: Just translate from scratch.
    """
    global CACHE
    source_lang = lang2name[line["source_lang"]]
    target_lang = lang2name[line["target_lang"]]
    source_text = line["source_segment"]

    # cache guard
    if (source_text, source_lang, target_lang) in CACHE["baseline1"]:
        print("Using cache!")
        return CACHE["baseline1"][(source_text, source_lang, target_lang)]

    try:
        response = client.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            prompt=f"Translate the following from {source_lang} to {target_lang}. Include only the translation (without the ''') and nothing else.\n'''{source_text}'''",
            max_tokens=len(source_text.split()) * 3
        )
        text = response.choices[0].text.strip().strip("'").replace("'''", "")
        text = text.replace("\n", "\\n")
    except Exception as e:
        print(f"Error: {e}")
        text = None


    CACHE["baseline1"][(source_text, source_lang, target_lang)] = text
    return text

OUTPUT_BASELINE1 = []
for line_i, line in enumerate(tqdm.tqdm(data)):
    OUTPUT_BASELINE1.append(translate_baseline1(line))

# baseline0: no post-edit
# baseline: post-edit without QE
# baseline: post-edit with QE
# baseline: post-edit with QE iteratively

# %%
from typing import List, Tuple, Any
import os
import copy

def dump_output(
    name: str,
    description: str,
    output: List[str],
):
    os.makedirs(f"outputs/{name}", exist_ok=True)
    if os.path.exists(f"outputs/{name}/{name}.zip"):
        # remove
        os.remove(f"outputs/{name}/{name}.zip")

    with open(f"outputs/{name}/metadata.txt", "w") as f:
        print("baseline organizers", file=f)
        print(description, file=f)

    data_new = copy.deepcopy(data)
    for line, segment in zip(data_new, output):
        if segment is None or segment == "":
            print("Falling back to hypothesis_segment")
            line["post_edit"] = line["hypothesis_segment"]
        else:
            line["post_edit"] = segment
    
    with open(f"outputs/{name}/predictions.tsv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=data_new[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(data_new)

    # zip predictions and metadata without directory structure
    import zipfile
    with zipfile.ZipFile(
        file=f"outputs/{name}/{name}.zip",
        mode="w",
        compression=zipfile.ZIP_BZIP2,
    ) as zf:
        zf.write(f"outputs/{name}/metadata.txt", arcname="metadata.txt")
        zf.write(f"outputs/{name}/predictions.tsv", arcname="predictions.tsv")



dump_output("baseline0", "No post-editing baseline. Uses gpt-4.1-nano-2025-04-14.", OUTPUT_BASELINE0)
dump_output("baseline1", "Translate from scratch baseline. Uses gpt-4.1-nano-2025-04-14.", OUTPUT_BASELINE1)

# %%

import pickle
with open("cache.pkl", "wb") as f:
    pickle.dump(dict(CACHE), f)

# %%

import pickle
with open("cache.pkl", "rb") as f:
    CACHE = pickle.load(f)