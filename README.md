# KeyVideoLLM

# VideoLLMs Experiment Setup

This repository contains the setup and requirements for conducting KeyVideoLLM.

## Installation

To set up the environment, you will need to install the following dependencies. The specified versions ensure compatibility and performance.

### Dependencies

```bash
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.5.3.56
pip install transformers==4.6.1
pip install ujson
pip install pandas
pip install ftfy
pip install tensorboard
pip install scipy
```
Alternatively, you can install all dependencies at once by using the requirements.txt file:

# VideoLLMs Experiments

## Installation and Setup

After installing the dependencies, you can proceed with your VideoLLMs experiments. We use the pre-trained CLIP model with a patch size of 32 as the encoder for keyframe selection due to its superior performance in aligning visual and textual data.

### Downloading the CLIP Model

Before running the code, download the CLIP model from Hugging Face. Specifically, we use `clip-vit-base-patch32`, which can be downloaded from the following link: [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32).

### Paths to Configure

Ensure you are aware of the following paths and their purposes:

- `--videos_dir`: Path to the directory containing the videos to be extracted.
- `--videos_json`: Path to the JSON file containing `video_id`, `question`, and `output_video_id`.
- `--output_dir`: Path where the extracted videos will be stored.
- `--num_frames`: Number of coarse frames to extract.

### Running the Code

Below is an example command to run the inference:

```bash
CUDA_VISIBLE_DEVICES=7 python inference.py --exp_name=exp_out --videos_start=1000 --videos_end=1001 --batch_size=1 --huggingface --load_epoch=-1
