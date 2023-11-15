from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Load the PLIP model
model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")

# Specify the images / directory containing image
image_path = "test.jpg"
image = Image.open(image_path)

candidate_labels = ["benign", "malignant"]

inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # This is the similarity score
probs = logits_per_image.softmax(dim=1)  
print(probs)
