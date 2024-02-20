import torch
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_path = 'dataset_cache'):
        super().__init__()
        dir_path = Path(dir_path)
        self.paths = list(dir_path.glob('*'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wf, f0, hubert_features, spk_id = torch.load(self.paths[idx])
        return wf, f0, hubert_features, spk_id
