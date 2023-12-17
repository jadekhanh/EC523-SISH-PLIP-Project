import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import openslide
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd

class Quantize(nn.Module):
    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size,1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        _, _ = z.shape  # z has shape (1, 512)
        weight = self.embedding.weight

        flat_inputs = z.view(-1, self.code_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.max(-distances, dim=1)[1]
        quantized = self.embedding(encoding_indices)
      
        return quantized, encoding_indices
    

# Load the pretrained PLIP model
plip_model = CLIPModel.from_pretrained("vinid/plip")
plip_processor = CLIPProcessor.from_pretrained("vinid/plip")

# Define quantization layer
quantization_layer = Quantize(128, 512)
# We have 128 vectors
# Each vector has dimension 512

# Use the same transform as original
transform_vqvqe = transforms.Compose([transforms.Lambda(lambda x: 2 * transforms.ToTensor()(x) - 1)])

top_directory = "/projectnb/ec523kb/projects/Pathology Image Search/SISH/SISH_patch/SISH/DATA_PATCH/kather100k_dataset_test/kather100k_dataset_ptif"

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(quantization_layer.parameters(), lr=0.1)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# Fine-tune the codebook
num_epochs = 5
loss_df = pd.DataFrame(columns=["Epoch", "Loss"])
total_files = sum([len(files) for _, _, files in os.walk(top_directory)])

for epoch in range(num_epochs):
    with tqdm(total=total_files, desc=f"Epoch {epoch + 1}", leave=False, dynamic_ncols=True) as epoch_iterator:
        epoch_loss = 0.0  

        print(f"Epoch {epoch + 1}")

        for root, dirs, files in os.walk(top_directory):
            for file in files:
                if file.lower().endswith('.tif'):
                    image_full_path = os.path.join(root, file)
                    print(image_full_path)
                    # Extract the patch name (file name without extension) and ground truth
                    patch_name = os.path.splitext(file)[0]
                    ground_truth = patch_name.split("-")[0]

                    optimizer.zero_grad()

                    patch = openslide.OpenSlide(image_full_path)
                    patch_rescaled = patch.read_region((0, 0), 0, (224, 224)).convert('RGB').resize((1024, 1024))


                    inp = transform_vqvqe(patch_rescaled)
                    inp = torch.unsqueeze(inp, 0) # After unsqueeze: [1, 3, 1024, 1024]
                    inp = inp.to(device, non_blocking=True)

                    inp = torch.squeeze(inp, 0)

                    candidate_labels = ["adipose tissue", "background", "debris", "lymphocytes", "mucus", "smooth muscle", "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]

                    input_processing = plip_processor(text=candidate_labels, images=inp, return_tensors="pt", padding=True)
                    vision_output = plip_model(**input_processing)
                    embeddings = vision_output['image_embeds']


                    # Apply quantization 
                    quantized_embeddings, indices = quantization_layer(embeddings)

                    loss = criterion(quantized_embeddings, embeddings)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    epoch_iterator.set_postfix({"Loss": loss.item()}, refresh=True)    
                    epoch_iterator.update(1)

    average_epoch_loss = epoch_loss / total_files
    print(f"Epoch {epoch + 1}, Average Loss: {average_epoch_loss}")
    loss_df = loss_df.append({"Epoch": epoch + 1, "Loss": average_epoch_loss}, ignore_index=True)

# Save the fine-tuned quantization layer
torch.save(quantization_layer.state_dict(), 'finetuned_codebook_semantic.pth')
loss_df.to_csv('losses_final', index=False)

epoch_iterator.close()
