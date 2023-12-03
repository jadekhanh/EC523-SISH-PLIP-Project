from PIL import Image
import os
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Load the CLIP model
model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")

# Specify the top-level directory
top_directory = "/projectnb/ec523kb/projects/Pathology Image Search/SISH/SISH_patch/SISH/DATA_PATCH/All/kather100k"

# Initialize an empty list to store DataFrames
dfs_list = []

# Get the total number of files to process
total_files = sum([len(files) for _, _, files in os.walk(top_directory)])

# Initialize the tqdm progress bar
pbar = tqdm(total=total_files, desc="Processing files")

# Loop through all subdirectories and files
for root, dirs, files in os.walk(top_directory):
    for file in files:
        # Check if the file has a ".tif" extension
        if file.endswith(".tif"):
            # Construct the full path to the image
            image_full_path = os.path.join(root, file)
            parts = file.split("-")
            label = parts[0]

            # Open the image using PIL
            image = Image.open(image_full_path)

            # Perform zero-shot classification
            candidate_labels = ["adipose tissue", "background", "debris", "lymphocytes", "mucus", "smooth muscle", "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]

            # Use CLIP processor to encode text and image
            inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

            # Use CLIP model to get logits and probabilities
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  

            # Get the predicted label
            predicted_label_index = torch.argmax(probs)
            predicted_label = candidate_labels[predicted_label_index]

            # Create a DataFrame for the current iteration
            df_temp = pd.DataFrame({"File": [file], "Label": [label], "Predicted_Label": [predicted_label], "Probabilities": [probs.tolist()]})

            # Append the DataFrame to the list
            dfs_list.append(df_temp)

            # Update the progress bar
            pbar.update(1)

# Close the progress bar
pbar.close()

# Concatenate all DataFrames in the list
df = pd.concat(dfs_list, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv("output_predictions.csv", index=False)
