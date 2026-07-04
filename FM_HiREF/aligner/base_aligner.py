from abc import ABC, abstractmethod

class BaseAligner(ABC):
    def __init__(self, alignment_threshold, aligner):
        self.alignment_threshold = alignment_threshold
        self.aligner = aligner  
        
        self.base_task_data = []

    @abstractmethod
    def load_dataset(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def begin_alignment(self, seed):
        raise NotImplementedError

    @abstractmethod
    def fetch_aligned_noise(self):
        raise NotImplementedError