# Implicit-Semantic-Response-Alignment
Pytorch implementation for paper **Implicit Semantic Response Alignment**.

# Abstract
Unsupervised domain adaptation (DA) transfers knowledge from a labeled source domain to an unlabelled target domain that is related but different from the source. Many research efforts have been devoted to solving various DA problems, including partial domain  adaptation (PDA) where the source label space submerges the target label space. Most current PDA methods address the mismatched label space by re-weighting the training categories or samples, to eliminate the influence of those extra classes. However, we believe the irrelevant categories also contain information that could benefit positive knowledge transfer. In this paper, we propose the Implicit Semantic Response Alignment (ISRA), an add-on module to existing DA methods, to uncover the inter-category relationships with a novel feature-level weighting schema. Specifically, our method first utilizes a class2vec machine to unsheathe implicit semantic topics and then calculates the features’ semantic response to each individual topic with an attention layer. Finally, our model aligns the attention responses across both domains and assigns weights to features, instead of samples, to retain the valuable information shared by multiple categories. We conduct comprehensive experiments on different DA benchmark datasets, as well as detailed in-depth analyses, to illustrate the effectiveness of our proposed ISRA model over the state-of-art.

# Prerequisites
- python == 3.6.8
- pytorch ==1.1.0
- orchvision == 0.3.0
- numpy, scipy, PIL, argparse, tqdm, pandas

# Framework
![Alt text](framework.png?raw=true "Title")

# Datasets
The datasets are set up with the same data protocol as [PADA](https://github.com/thuml/PADA/tree/master/pytorch/data).
Please download [Office31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) and [ImageNet-Caltech](https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view) datasets and change the path in the image list files (*.txt) in the './data/' directory.

# Running
Run the code for PDA on Office-Home for Task (Ar -> Cl)
> python run_partial_with_centroids.py --s 0 --t 1 --dset office_home --net ResNet50 --cot_weight 1. --output run1 --gpu_id 0 --rcwt 1. --alwt 1. --ctr_weight 1. --fdim 512 --edim 256 

# Acknowledgement
This project is built on the open-source implementation [BA3US](https://github.com/tim-learn/BA3US)