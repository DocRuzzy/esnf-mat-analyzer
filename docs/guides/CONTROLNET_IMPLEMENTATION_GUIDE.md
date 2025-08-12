# ControlNet Integration Guide

This document provides instructions for integrating ControlNet into the synthetic data pipeline for refining ruler images.

## 1. Download the Pre-trained Model

The first step is to download a pre-trained ControlNet model. The recommended model for this task is `control_sd15_scribble.pth`, which is trained on boundary edges and is well-suited for refining synthetic ruler images.

The model can be downloaded from the official ControlNet Hugging Face repository:
[https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth](https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth)

It is recommended to save the model in a `models` directory in the root of the project.

**Note**: The model is large (~5.7 GB), so ensure you have enough disk space.

## 2. Implement the Refinement Pipeline

The next step is to implement a pipeline that takes a synthetic ruler image and uses ControlNet to refine it. This will involve the following steps:

1.  **Load the ControlNet model**: Load the downloaded `control_sd15_scribble.pth` model using a library like `torch`.
2.  **Load a Stable Diffusion model**: ControlNet works with a pre-trained Stable Diffusion model. You will need to load a Stable Diffusion model (e.g., `stable-diffusion-v1-5`) from the Hugging Face Hub.
3.  **Create a ControlNet pipeline**: Use the `StableDiffusionControlNetPipeline` from the `diffusers` library to create a pipeline that combines the Stable Diffusion model and the ControlNet model.
4.  **Process the synthetic image**: The synthetic ruler image needs to be processed to be used as a condition for the ControlNet model. This will likely involve converting it to a PIL image and resizing it.
5.  **Run the pipeline**: Run the pipeline with the processed synthetic image and a text prompt (e.g., "a ruler on a flat surface"). The pipeline will return a refined, more realistic ruler image.

The code for this pipeline can be adapted from the Gradio demos in the official ControlNet repository:
[https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)

## 3. Integrate into the Synthetic Data Generator

Once the refinement pipeline is implemented, it should be integrated into the `SyntheticRulerGenerator` class in `esnf_mat_analyzer/validation/synthetic_ruler_generator.py`.

A new method should be added to the class that takes a generated ruler image and applies the ControlNet refinement to it. This will allow for easy generation of refined synthetic rulers for training and validation.

## 4. Update the PyTorch Dataset

Finally, the `SyntheticDataset` in `esnf_mat_analyzer/data/synthetic_dataset.py` should be updated to use the new refinement method. A new flag should be added to the dataset's `__init__` method to control whether to apply the ControlNet refinement or not. This will allow for easy comparison of the model's performance with and without the refinement.
