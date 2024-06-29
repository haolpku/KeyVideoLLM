# KeyVideoLLM

# VideoLLMs Experiment Setup

This repository contains the setup and requirements for conducting VideoLLMs experiments using the state-of-the-art framework, Videollava~\cite{video-llava}.

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

Usage
After installing the dependencies, you can proceed with your VideoLLMs experiments. The pre-trained CLIP model with a patch size of 32 will be used as the encoder for keyframe selection due to its superior performance in aligning visual and textual data.

For detailed instructions on running experiments, please refer to the experiment scripts and documentation within the repository.

需要的clip 模型:hf上下载clip-vit-base-patch32
1、
model/clip_baseline.py
self.clip = CLIPModel.from_pretrained("/data/wentao/clip-vit-base-patch32/")
2、inference.py
tokenizer = CLIPTokenizer.from_pretrained("/data/wentao/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

需要修改的路径：
1、输出的视频文件
trainer/trainer.py
dir='/mnt/share/video/video_dl_space/xpool_clip_keyframe/keyframe_clip_videos/videochatgpt_tune/'

2、原视频路径、json文件路径、处理视频的个数，初次均匀抽帧的个数
config/all_config.py
parser.add_argument('--videos_dir', type=str, default="/data/wentao/lijp_data/video_llava_data/train_data/", help="Location of videos")
parser.add_argument('--videos_json', type=str, default='/data/wentao/lijp_data/video_llava_data/train_data/train_json/videochatgpt_tune_.json', help="Location of videos") d
#parser.add_argument('--msrvtt_train_file', type=str, default='9k')
parser.add_argument('--videos_num_read', type=int, default=10) #处理的视频个数，-1为全部视频都处理!
parser.add_argument('--num_frames', type=int, default=32) #初次抽帧数

执行代码：
CUDA_VISIBLE_DEVICES=7 python inference.py --exp_name=exp_out  --videos_start=1000 --videos_end 1001 --batch_size=1 --huggingface --load_epoch=-1
