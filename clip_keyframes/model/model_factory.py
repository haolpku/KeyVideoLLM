from config.base_config import Config
from model.clip_extract import CLIPExtract


class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'clip_extract':
            return CLIPExtract(config)
        else:
            raise NotImplemented
