conda create --name keyvideollm python=3.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install transformers==4.6.1
pip install ujson
pip install pandas
pip install ftfy


CUDA_VISIBLE_DEVICES=0 nohup python inference.py --exp_name=exp_out  --videos_start=10000 --videos_end=20000 --batch_size=1 --huggingface --load_epoch=-1 > outputs/keyframe_clip_videos_sort/test1.log 2>&1 &
