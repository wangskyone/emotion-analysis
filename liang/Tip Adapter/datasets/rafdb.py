import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

from .oxford_pets import OxfordPets

import torch
torch.autograd.detect_anomaly(True)

# template = ['The expression is {}.']
# template = ['The expression of the person in the picture is {}.']
template = ['{}']

# TODO:要将数据集做适配
class Rafdb(DatasetBase):
    dataset_dir = 'rafdb'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'rafdb.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
