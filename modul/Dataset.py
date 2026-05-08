import torch
from torch.utils.data import Dataset

def collate_fn(batch):
    return list(zip(*batch))

class DrivingDataset(Dataset):
    def __init__(self, samples, return_scene_info=False):
        self.samples = samples
        self.return_scene_info = return_scene_info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        hist_motion = torch.tensor(s["hist_motion"], dtype=torch.float32)
        style_long = torch.tensor(s["style_long"], dtype=torch.float32).mean(dim=0)
        style_lat = torch.tensor(s["style_lat"], dtype=torch.float32).mean(dim=0)
        intent = torch.tensor(s["intent"], dtype=torch.long)
        future = torch.tensor(s["future"], dtype=torch.float32)
        interaction = torch.tensor(s["interaction"], dtype=torch.float32)
        last_real  = torch.tensor(s["hist_last"], dtype=torch.float32)

        if self.return_scene_info:
            scene_info = {
                'scene_id': s.get('scene_id', 'unknown'),
                'file_num': s.get('file_num', -1)
            }
            return hist_motion, style_long, style_lat, intent, future, interaction, last_real, scene_info
        else:
            return hist_motion, style_long, style_lat, intent, future, interaction, last_real


