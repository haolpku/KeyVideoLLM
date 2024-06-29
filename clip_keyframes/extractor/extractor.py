from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from clip_keyframes.extractor.base_extractor import BaseExtractor
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from tqdm import tqdm
import cv2
import os

def write_video(frames_ids_list,video_path,conversations_id,output_dir):
    conversations_id=int(conversations_id.cpu())
    video_path_str=str(video_path[0])
    #print('video_path is:',video_path_str)
    base_video=os.path.basename(video_path_str)
    ext = os.path.splitext(base_video)[1].lower()

    dir=output_dir
    output_file_name = dir+str(conversations_id)+'.mp4'
    #output_file_name = dir+os.path.splitext(base_video)[0]+'_'+str(conversations_id)+'.mp4'
    #print('output_file_name is:',output_file_name)

    if os.path.exists(output_file_name):
        print(f'{output_file_name} already exists, skipping...')
        return 
    #print('Writing  video ...')

    # load original video
    cap = cv2.VideoCapture(video_path_str)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create summary video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.mkv':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))
    vlen=0
    for index in frames_ids_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ret, frame = cap.read()
        
        if ret:
            out.write(frame)
            vlen+=1
    if vlen<len(frames_ids_list):
        print(output_file_name+'_vlen is:',vlen)
    #print(output_file_name+'_vlen is:',vlen)
    out.release()
    cap.release()

  
class Extractor(BaseExtractor):
    """
    Extractor class
    Note:
        Inherited from BaseExtractor.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.extract_type = config.extract_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0
        self.output_dir=config.output_dir


    

    
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)
                
                text_embed, vid_embed, vid_embed_pooled,frames_ids_list,video_path,conversations_id = self.model(data, return_all_frames=True)
                write_video(frames_ids_list,video_path,conversations_id,self.output_dir)

            return 0
