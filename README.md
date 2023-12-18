# Pathology Image Search: A Comparative Analysis of Self-Supervised Algorithms and Zero-Shot Transfer with Visual Language Encoders for Patch Retrieval 

## Collaborators
Phuong Khanh Tran, Susan Zhang, Daniellia Sumigar, Priscila Rubio

## Tasks
Our project's primary objective is to evaluate the performance of PLIP and MI-Zero multimodal vision-language models within the context of zero-shot classification. Should PLIP demonstrate superior performance over MI-Zero, we plan to strategically replace the SISH image encoder with the PLIP image encoder. Conversely, if MI-Zero yields better results, we would consider a similar replacement using the MI-Zero image encoder. Furthermore, our project seeks to potentially enhance SISH's image retrieval capabilities by introducing vision-language understanding into its existing pipeline.

## SISH Image Encoding
SISH utilizes two different image encoders: (1) a Vector Quantized-Variational AutoEncoder (VQ-VAE), which is a type of variational autoencoder that uses vector quantization to learn discrete, latent codes, and (2) DenseNet121, a deep convolutional network, pre-trained on ImageNet. 
VQ-VAE maps input patches of WSIs to a latent space. The discrete latent representation of the patches is then fed into a series of average pooling operations to generate integer indices. DenseNet121 is used along with binarization to produce a binary string of each input patch for feature extraction. Both these indices and binarized features are sent into the SISH search and ranking algorithm to determine the candidate patches.

## MI-Zero Image Encoding
MI-Zero employs openly accessible text encoders, specifically BioClinicalBert and PubMedBert, which have been trained on biomedical and clinical datasets, specifically PubMed and MIMIC. As the default choice for the image encoder, MI-Zero employs a state-of-the-art CTransPath encoder, which has been trained using self-supervised representation learning on unlabeled 15.5 million histopathology patches and has consistently demonstrated remarkable superiority over features initialized from ImageNet, across various downstream tasks.

## PLIP Image Encoding
PLIP is a model trained on image-text pairings provided by the OpenPath dataset. It follows a model architecture similar to Vit-B-32, which is widely used for image classification tasks. The encoder determines image embedding vectors from a given input image patch, and the similarity scores of these input embedding vectors are calculated along with the target image embeddings for classification.

## Zero-Shot Classification
We conducted an evaluation of the image encoders of PLIP and MI-Zero models in the context of zero-shot classification, which involves categorizing previously unseen examples belonging to classes not encountered during training. In this process, a vision-language model is employed, where images are passed through the image encoder, and class prompts are directed to the text encoder. Subsequently, the resulting embeddings are compared within a shared latent space, and the class with the highest score is assigned as the predicted label for the input image. This approach serves as a valuable benchmark for assessing the performance of image retrieval systems. 

## Improving Experiment on Patch Retrieval
Improving patch retrieval accuracy necessitates a comparative analysis of zero-shot classification outcomes between PLIP and MI-Zero. The model yielding the highest accuracy in this task would be chosen, and its image encoder would replace the existing SISH image encoder. Following this substitution, we evaluated the baseline performance accuracy and precision of the enhanced SISH, aiming to determine if there was an improvement in patch retrieval capabilities. We hoped that extending SISH to accept multimodal queries, such as text or genomic data, could offer an efficient method for identifying semantic similarities across diverse and distinct data types, enabling researchers to compare and retrieve relevant information beyond visual content.

## Dataset
The evaluation was conducted on a comprehensive dataset of histological images from the Kather Colon dataset (available at https://zenodo.org/record/1214456) consisting of 100,000 image patches extracted from 86 colorectal cancer tissue slides originally utilized for overall colorectal cancer survival prediction. It encompasses a diverse array of nine distinct tissue types, including tumor epithelium (TUM), cancer-related stroma (STR), smooth muscle (MUS), immune cell conglomerates (LYM), debris/necrosis (DEB), mucus (MUC), normal colon mucosa (NORM), adipose tissue (ADI), and background (BACK). The dataset is designed to be approximately balanced and features images of dimensions 224 × 224 with a resolution of 122µ/px.

## References
1. C. Chen, M. Y. Lu, D. F. K. Williamson, T. Y. Chen, A. J. Schaumberg, and F. Mahmood, “Fast and scalable search of whole-slide images via self-supervised deep learning,” Nature Biomedical Engineering, vol. 6, no. 12, pp. 1420–1434, Dec. 2022, doi: https://doi.org/10.1038/s41551-022-00929-8. 
2. M. Lu et al., “Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images,” arXiv (Cornell University), Jun. 2023, doi: https://doi.org/10.48550/arxiv.2306.07831. 
3. Z. Huang, F. Bianchi, M. Yuksekgonul, T. J. Montine, and J. Zou, “A Visual–Language Foundation Model for Pathology Image Analysis using Medical Twitter,” Nature Medicine, vol. 29, no. 9, pp. 2307–2316, Sep. 2023, doi: https://doi.org/10.1038/s41591-023-02504-3. 
4. S. Kalra et al., “Pan-cancer Diagnostic Consensus through searching Archival Histopathology Images using Artificial Intelligence,” npj Digital Medicine, vol. 3, no. 1, pp. 1–15, Mar. 2020, doi: https://doi.org/10.1038/s41746-020-0238-2.
5. S. Kalra, C. Choi, S. Shah, L. Pantanowitz, and H. R. Tizhoosh, “Yottixel -- An Image Search Engine for Large Archives of Histopathology Whole Slide Images,” arXiv (Cornell University), Nov. 2019, doi: https://doi.org/10.48550/arxiv.1911.08748. 
6. A. Galdran, K. Hewitt, N. Ghaffari, J. Kather, G. Carneiro, and M. González Ballester, “Test Time Transform Prediction for Open Set Histopathological Image Recognition,” Jun. 2022. Accessed: Dec. 14, 2023. [Online]. Available: https://arxiv.org/pdf/2206.10033.pdf
7. Microscopes International, LLC. “General Interest: What Are Tiled Tiff (SVS) Files?” Microscopes International Corporate, 16 July 2019, www.microscopesinternational.com/support/kb/article/ngn1266.aspx. 

## Github
* SISH: https://github.com/mahmoodlab/SISH
* MI-Zero: https://github.com/mahmoodlab/MI-Zero 
* PLIP: https://huggingface.co/spaces/vinid/webplip
