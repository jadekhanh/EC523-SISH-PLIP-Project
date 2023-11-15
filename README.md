# Comparative Analysis of Self Supervised Image Search for Histology and Pathology Language-Image Pretraining for Zero-Shot Classification in Rare Disease Subtype Retrieval

## Collaborators
Phuong Tran, Susan Zhang, Daniellia Sumigar, Priscila Rubio

## Tasks
To assess the comparative performance of self-supervised image search for histology (SISH) and pathology language-image pretraining (PLIP), their effectiveness within the context of zero-shot classification will be evaluated. If SISH proves to outperform PLIP, a strategic replacement of the image encoder in PLIP will be performed with the image encoder from SISH. However, should PLIP demonstrate better results, further experimentation will be undertaken to enhance the zero-shot classification of “Rare Disease Subtype Retrieval”.


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

## GITHUB
* SISH: https://github.com/mahmoodlab/SISH
* PLIP: https://huggingface.co/spaces/vinid/webplip
