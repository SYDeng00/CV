import os
from tqdm import tqdm
import argparse
import numpy as np  # Make sure to import numpy

# torch imports
import torch
import torch.nn as nn
from torch.optim import AdamW

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# custom imports
from model import Network
from loss import VFILoss

from dataset import TripletDataset
from dataset.helpers import *

#NOTE: DO NOT CHANGE THE SEEDs
torch.manual_seed(102910)
np.random.seed(102910)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 10

# rest of your setup code ...

"""
Training loop
"""
logger = SummaryWriter(os.path.join("runs", f'conf:{conf}'))
os.makedirs('runs', exist_ok=True)
os.makedirs('weights', exist_ok=True)
for epoch in tqdm(range(NUM_EPOCHS), desc=f'Training for conf {conf}'):
    model.train()
    total_train_loss = 0
    for f1, f2, f3 in train_loader:
        # TASK 2: Implement the training loop
        # Transfer data to the device
        f1, f2, f3 = f1.to(device), f2.to(device), f3.to(device)
        
        # Forward pass through the model
        output = model(f1, f2, f3)

        # Calculate loss
        # Note: You'll need to modify this if your loss_fn expects more parameters
        batch_loss = loss_fn(f1, f2, f3, output)

        # Zero the gradients before backward pass
        optimizer.zero_grad()

        # Backward pass
        batch_loss.backward()

        # Perform a single optimization step
        optimizer.step()

        # Update the training loss
        total_train_loss += batch_loss.item()
    
    # Average the loss over all batches and log it to TensorBoard
    total_train_loss /= len(train_loader)
    logger.add_scalar('Train loss', total_train_loss, global_step=epoch+1)

# Save the model after training
torch.save(model.state_dict(), os.path.join('weights', f'conf-{conf}.pth'))
