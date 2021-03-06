from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.Singleton import Singleton
from future.utils import with_metaclass


class DataSetStore(with_metaclass(Singleton)):
    def __init__(self, dataset_info):
        if not hasattr(self, 'dataset_info'):
            self.dataset_info = YAMLLoader.read_yaml(dataset_info)
