# DA241M Minor Project: Semantic Segmentation of Indian Roadways

This project performs semantic segmentation on a dataset of Indian roadways under various conditions. The task involves segmenting diverse and complex roadway scenes, which include multiple object sizes and intricate visual details.

## Model Overview
Our main model for this task is [Mask2Former](https://arxiv.org/abs/2112.01527), a universal segmentation model capable of performing instance, semantic, and panoptic segmentation. Mask2Former achieves robust performance through several key innovations:

- *Masked Attention*: Instead of traditional cross-attention, it utilizes masked attention to localize feature focus around predicted segments, leading to faster convergence and improved segmentation accuracy.
- *Multi-Scale High-Resolution Features*: Effectively segments objects of various sizes by leveraging multi-scale, high-resolution features.
- *Dynamic Mask Prediction*: Predicts dynamic masks rather than per-pixel labels, providing adaptability for complex segmentation tasks.

For finetuning, we froze the encoder backbone and pixel decoder to preserve learned features and finetuned only the transformer decoder and MLP layer, making training more resource-efficient.

## Dataset Preparation
The training dataset initially consisted of 8,000 high-resolution images, but due to computational constraints, we implemented a filtering process to extract the most informative subset of images for finetuning. Our approach used a weighted score system to rank images based on specific criteria:

- *Class Diversity Score (CDS)*: Measures the diversity of classes in an image.
- *Rare Class Score (RCS)*: Scores images based on the presence of rare classes.
- *Image Count Score (ICS)*: Counts images in each subdirectory.

Using these metrics, a weighted score was calculated as follows:

markdown
Weighted Score = α * (CDS / max(CDS)) + β * (RCS / max(RCS)) + γ * (ICS / max(ICS))

where *α = 0.4, **β = 0.4, and **γ = 0.2*.

Using this ranking, we selected 1,500 high-value images due to computational constraints.

This dataset is hosted on Kaggle and can be accessed via the following link:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/shayakbhattacharya/finetune)

## Model and Training Settings

After preparing the dataset, we used Hugging Face (HF) modules to build the finetuning pipeline. The dataset was converted to the Mask2Former format through a custom pipeline based on the HF dataset structure. This included a custom collate function that returns 6 key items for each image:

- *pixel_values*: Image as a numpy array after transformations
- *pixel_mask*: Regions of the segmentation map to attend to
- *mask_labels*: N masks for objects within the image
- *class_labels*: N class labels for the image objects
- *original_images*: Untransformed images
- *original_segmentation_maps*: Unaltered segmentation maps

Using the Mask2Former preprocessor, each image was further processed into the required model format. We modified the final output layer based on the number of classes in our dataset.

### Baseline Hyperparameters

- *Train/Val/Test Split*: 85.5% / 4.5% / 10%
- *Epochs*: 2
- *Batch Size*: 8
- *Optimizer*: Adam
- *Learning Rate*: 5e-5

## Pipeline
The segmentation pipeline integrates:
- *Mask2Former* as the primary model
- Efficient data processing and inference pipelines to maximize segmentation accuracy and minimize computational overhead.
