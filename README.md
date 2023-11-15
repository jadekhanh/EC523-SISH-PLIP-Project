# Comparative Analysis of Self Supervised Image Search for Histology and Pathology Language-Image Pretraining for Zero-Shot Classification in Rare Disease Subtype Retrieval

## Collaborators
Phuong Khanh Tran, Susan Zhang, Daniellia Sumigar, Priscila Rubio

## Tasks
To assess the comparative performance of self-supervised image search for histology (SISH) and pathology language-image pretraining (PLIP), their effectiveness within the context of zero-shot classification will be evaluated. If SISH proves to outperform PLIP, a strategic replacement of the image encoder in PLIP will be performed with the image encoder from SISH. However, should PLIP demonstrate better results, further experimentation will be undertaken to enhance the zero-shot classification of “Rare Disease Subtype Retrieval”.

## SISH Image Encoding
SISH utilizes two different image encoders: (1) a Vector Quantized-Variational AutoEncoder (VQ-VAE), which is a type of variational autoencoder that uses vector quantization to learn discrete, latent codes, and (2) DenseNet121, a deep convolutional network, pre-trained on ImageNet. 
VQ-VAE maps input patches of WSIs to a latent space. The discrete latent representation of the patches is then fed into a series of average pooling operations to generate integer indices. DenseNet121 is used along with binarization to produce a binary string of each input patch for feature extraction. Both these indices and binarized features are sent into the SISH search and ranking algorithm to determine the candidate patches.

## PLIP Image Encoding
PLIP is a model trained on image-text pairings provided by the OpenPath dataset. It follows a model architecture similar to Vit-B-32, which is widely used for image classification tasks. The encoder determines image embedding vectors from a given input image patch, and the similarity scores of these input embedding vectors are calculated along with the target image embeddings for classification.

## Zero-shot Classification
We will assess the performance of both the image encoders on zero-shot classification. Zero-shot classification describes the task of classifying unseen examples belonging to classes not present during training. For image retrieval systems, the ability to perform zero-shot classification provides a solid benchmark for performance evaluation. As outlined in [3], we will replicate the procedure on the same four selected datasets provided by Kather Colon, PanNuke, DigestPath, and WSSS4LUAD. 

## Improving Experiment on Rare Disease Subtype Retrieval
The ‘Rare Disease Subtype Retrieval’ experiment outlined in [1] describes the task of inputting patch images of rare diseases with limited image data. Based on the result of the previous zero-shot classification task, we will apply the better-performing image encoder to the ‘Rare Disease Subtype Retrieval’ experiment. We plan to utilize the dataset supplied by Brigham and Women’s Hospital (BWH) and The Cancer Genome Atlas (TCGA), just as it was employed in [1]. 

## Dataset
* TCGA: https://portal.gdc.cancer.gov
* Kather colon: https://zenodo.org/record/1214456 
* PanNuke: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke 
* DigestPath: https://digestpath2019.grand-challenge.org/ 
* WSSS4LUAD: https://wsss4luad.grand-challenge.org/ 

## References
1. Chen, C., Lu, M.Y., Williamson, D.F.K. et al. Fast and scalable search of whole-slide images via self-supervised deep learning. Nat. Biomed. Eng 6, 1420–1434 (2022). https://doi.org/10.1038/s41551-022-00929-8
2. Huang, Z., Bianchi, F., Yuksekgonul, M. et al. A visual–language foundation model for pathology image analysis using medical Twitter. Nat Med 29, 2307–2316 (2023). https://doi.org/10.1038/s41591-023-02504-3 
3. Hegde, N. G. et al. Similar image search for histopathology: Smily. npj Digit. Med. 2, 56 (2019).
4. Kalra, S. et al. Pan-cancer diagnostic consensus through searching archival histopathology images using artificial intelligence. npj Dig. Med. 3, 31 (2020).
5. Kalra, S. et al. Yottixel–an image search engine for large archives of histopathology whole slide images. Med. Image Anal. 65, 101757 (2020).

## Github
* SISH: https://github.com/mahmoodlab/SISH
* PLIP: https://huggingface.co/spaces/vinid/webplip
