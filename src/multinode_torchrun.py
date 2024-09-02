import numpy as np
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



nb_gpus = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datapath = os.path.expandvars('/scratch/vp91/$USER/intro-to-pytorch/data/pima-indians-diabetes.data.csv')


# Define the custom Dataset class
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Define the custom Dataset class
class PimaDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file without header and assign column names
        self.data = pd.read_csv(csv_file, header=None, names=column_names)
        self.features = self.data.drop('Outcome', axis=1).values
        self.labels = self.data['Outcome'].values
        
        # Convert to PyTorch tensors
        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)
        
        # Calculate mean and std
        self.mean = self.features_tensor.mean(dim=0)
        self.std = self.features_tensor.std(dim=0)
        
        # Normalize the features
        self.features_tensor = (self.features_tensor - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.features_tensor[idx]
        label = self.labels_tensor[idx]
        return feature, label

def setup():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    
def cleanup():
    dist.destroy_process_group()

def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = PimaDataset(datapath)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
    

def main():

    # setup the process groups
    setup()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # prepare the dataloader
    dataloader = prepare(rank, world_size)
    
    # instantiate the model(it's your own model) and move it to the right device
    model = PimaClassifier().to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    for epoch in range(n_epochs):

        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)

        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(rank)
            batch_labels = batch_labels.to(rank)

            optimizer.zero_grad()
        
            outputs = model(batch_features)
        
            batch_labels = batch_labels.unsqueeze(1).float()
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    cleanup()





if __name__ == '__main__':
    main()

        
