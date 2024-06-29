import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

"""
Inputs (assume L2 normalized)
    text_embeds: num_texts x embed_dim
    video_embeds: num_vids x num_frames x embed_dim
"""

class ExtractKeyframes(nn.Module):
    def __init__(self, extract_type, config: Config):
        super(ExtractKeyframes, self).__init__()
        
        assert extract_type is not None, \
                'Need to specify extraction type when using baseline model.'

        if extract_type == 'avg':
            self.extraction_func = self._avg_extraction
            print("Using average extraction")
        
        elif extract_type == 'topk':
            self.k = config.k
            assert self.k > 0
            self.extraction_func = self._topk_extraction
            print("Using top-{} frame extraction".format(self.k))
        
        else:
            raise NotImplementedError

    
    def _avg_extraction(self, text_embeds, video_embeds):
        """
        Pooling mean of frames

        Output
            video_embeds_pooled: num_vids x embed_dim
        """
        video_embeds_pooled = video_embeds.mean(dim=1)
        return video_embeds_pooled

    
    def _topk_extraction(self, text_embeds, video_embeds):
        """
        Pooling top-k frames for each video based on
        similarities with each text query
        
        Output
            video_embeds_pooled: num_vids x num_texts x embed_dim
        """
        num_texts, embed_dim = text_embeds.shape

        # num_vids x num_frames x num_texts
        sims = video_embeds @ text_embeds.t()
        sims_topk = torch.topk(sims, self.k, dim=1)[1]

        # Make format compatible with torch.gather
        video_embeds = video_embeds.unsqueeze(-1).expand(-1, -1, -1, num_texts)
        sims_topk = sims_topk.unsqueeze(2).expand(-1, -1, embed_dim, -1)

        # num_vids x k x embed_dim x num_texts
        video_embeds_topk = torch.gather(video_embeds, dim=1, index=sims_topk)
        
        # Top-k extraction => num_vids x embed_dim x num_texts
        video_embeds_pooled = video_embeds_topk.sum(dim=1)
        return video_embeds_pooled.permute(0,2,1),sims_topk


    def forward(self, text_embeds, video_embeds):
        return self.extraction_func(text_embeds, video_embeds)
