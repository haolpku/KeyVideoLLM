import cv2
import random
import numpy as np
import torch


class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path,
                               num_frames,
                               sample='rand'):
        """
            video_path: str/os.path
            num_frames: int - number of frames to sample
            sample: 'rand' | 'uniform' how to sample
            returns: frames: torch.tensor of stacked sampled video frames 
                             of dim (num_frames, C, H, W)
                     idxs: list(int) indices of where the frames where sampled
        """
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened()), video_path
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # get indexes of sampled frames
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []

        # ranges constructs equal spaced intervals (start, end)
        # we can either choose a random image in the interval with 'rand'
        # or choose the middle frame with 'uniform'
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        frame_idxs_real=[]
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                #cv2.imwrite(f'images/{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
                frame_idxs_real.append(index)
            # else:
            #     raise ValueError
            try:
                assert ret , f"wrong read frame: {ret}"

            except AssertionError as e:
                print(f"Skipping {video_path}: {e}")
                continue


        while len(frames) < num_frames:
            print(video_path+'_while len(frames) < num_frames')
            frames.append(frames[-1].clone())
            frame_idxs_real.append(frame_idxs_real[-1])
            
        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs_real
