from config.base_config import Config
from datasets.model_transforms import init_transform_dict

from torch.utils.data import DataLoader
from datasets.set_dataset import setDataset

class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        test_img_tfms = img_transforms['clip_test']

       
        if config.dataset_name == 'set':
            if split_type == 'test':
                dataset = setDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)
        else:
            raise NotImplementedError
