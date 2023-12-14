import os
import pandas as pd

# For creating csv file containing images & labels in the data folder

top_directory = "/projectnb/ec523kb/projects/Pathology Image Search/SISH/SISH_patch/SISH/DATA_PATCH/All/kather100k"

dfs = []

for root, dirs, files in os.walk(top_directory):
    for file in files:
        if file.lower().endswith('.tif'):
            image_full_path = os.path.join(root, file)

            patch_name = os.path.splitext(file)[0]
            ground_truth = patch_name.split("-")[0]

            current_df = pd.DataFrame({"patch": [patch_name], "ground_truth": [ground_truth]})

            dfs.append(current_df)

df = pd.concat(dfs, ignore_index=True)

df.to_csv("image_info.csv", index=False)
