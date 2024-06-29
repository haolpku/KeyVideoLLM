import torch
import torch.nn as nn
from config.base_config import Config
from modules.extract_keyframes import ExtractKeyframes

class CLIPExtract(nn.Module):
    def __init__(self, config: Config):
        super(CLIPExtract, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("/root/KeyVideoLLM/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        self.pool_frames = ExtractKeyframes(config.extract_type, config)


    def forward(self, data, return_all_frames=False):
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        frame_idx_first = data['frame_idx']
        video_path=data['video_path']
        conversations_id=data['conversations_id']

        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        
        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)

        video_features_pooled ,frames_ids= self.pool_frames(text_features, video_features)
        #print('frames_ids is:',frames_ids)
        frames_ids_list=[]
        frames_ids_direct_list=[]
        for i in range(frames_ids.shape[1]):
            frames_ids_direct_list.append(frames_ids[0, i, 0, 0])  
            tem=int(frames_ids[0, i, 0, 0].cpu())
            frames_ids_list.append(frame_idx_first[tem])
        if return_all_frames:
            return text_features, video_features, video_features_pooled,frames_ids_list,video_path,conversations_id

        return text_features, video_features_pooled,frames_ids_list,video_path,conversations_id
