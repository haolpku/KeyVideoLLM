import os
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
import json

class setDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
        Notes: for test split, we return one video, caption pair for each caption belonging to that video
               so when we run test inference for t2v task we simply average on all these pairs.
    """

    def __init__(self, config: Config, split_type = 'test', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        db_file_dir = config.videos_json #
        print(db_file_dir)

        self.read_n_video=config.videos_num_read
        with open(db_file_dir, 'r') as f:
            db_file = json.load(f)
            if self.read_n_video!=-1:
                #self.json_data = db_file[:self.read_n_video]
                self.json_data = db_file[config.videos_start:config.videos_end]
                print('config.videos_start is:',config.videos_start)
                print('config.videos_end is:',config.videos_end)
            else:

                self.json_data = db_file
        self.all_test_pairs = []
        for video_item in self.json_data:
            conversations_id = video_item["question_id"]
            video_path_ori = video_item["video_name"]
            video_path = self.find_video_path(video_path_ori)
            print(video_path)
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                assert (cap.isOpened()), "cap.is not Opened"
            except AssertionError as e:
                print(f"Skipping {video_path}: {e}")
                continue
            caption = video_item["question"]
            print(conversations_id)
            self.all_test_pairs.append([conversations_id, video_path,caption])
            #caption=''
            #if conversations[0]["value"] != "" and  conversations[0]["from"]=="human":
            #    caption = conversations[0]["value"][8:]  #<video>\n
            #if conversations[-1]["value"] != "" and  conversations[-1]["from"]=="gpt":
            #    caption = caption+' '+conversations[-1]["value"]
            #    self.all_test_pairs.append([conversations_id, video_path,caption])
                #print(caption)
            #else:
            #    print(video_path+'_caption is wrong!')

    def find_video_path(self, question_id):
        for filename in os.listdir(self.videos_dir):
            if filename.startswith(question_id):
                return os.path.join(self.videos_dir, filename)
        return None

    def __getitem__(self, index):
        if self.split_type == 'train':
            video_path, caption, video_id = self._get_vidpath_and_caption_by_index_train(index)
        else:
            conversations_id,video_path, caption = self._get_vidpath_and_caption_by_index_test(index)


        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)
        
        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        ret = {
            'conversations_id': conversations_id,
            'video': imgs,
            'text': caption,
            'frame_idx': idxs,
            'video_path': video_path
        }

        return ret


    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, caption = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return video_path, caption, vid

    def _get_vidpath_and_caption_by_index_test(self, index):
        conversations_id, video_path,caption = self.all_test_pairs[index]
        return conversations_id,video_path, caption

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)
