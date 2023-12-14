from PIL import Image
import os
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")

top_directory = "/projectnb/ec523kb/projects/Pathology Image Search/SISH/SISH_patch/SISH/DATA_PATCH/All/kather100k"

dfs_list = []

total_files = sum([len(files) for _, _, files in os.walk(top_directory)])

pbar = tqdm(total=total_files, desc="Processing files")

for root, dirs, files in os.walk(top_directory):
    for file in files:
        if file.endswith(".tif"):
            image_full_path = os.path.join(root, file)
            parts = file.split("-")
            label = parts[0]
            image = Image.open(image_full_path)
            candidate_labels = ["adipose tissue", "background", "debris", "lymphocytes", "mucus", "smooth muscle", "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]

            inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  

            predicted_label_index = torch.argmax(probs)
            predicted_label = candidate_labels[predicted_label_index]

            df_temp = pd.DataFrame({"File": [file], "Label": [label], "Predicted_Label": [predicted_label], "Probabilities": [probs.tolist()]})

            dfs_list.append(df_temp)

            pbar.update(1)

pbar.close()

df = pd.concat(dfs_list, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv("output_predictions.csv", index=False)
