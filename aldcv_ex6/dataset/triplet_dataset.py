import os
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Each triplet has its folder with the corresponding 3 images.
    Structure of the triplet dataset:
        triplet_<number>
            -   1.png
            -   2.png
            -   3.png
    """
    def __init__(self, triplets_path, transform=None):
        self.triplets_path = triplets_path
        self.transform = transform
        # Make sure to only include directories
        self.triplets = [d for d in os.listdir(triplets_path) if os.path.isdir(os.path.join(triplets_path, d))]

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, index):
        # Get the path for the current triplet
        current_triplet_path = os.path.join(self.triplets_path, self.triplets[index])
        
        # Load images
        frame_1 = Image.open(os.path.join(current_triplet_path, '1.png')).convert('RGB')
        frame_2 = Image.open(os.path.join(current_triplet_path, '2.png')).convert('RGB')
        frame_3 = Image.open(os.path.join(current_triplet_path, '3.png')).convert('RGB')

        # Apply transformations if any
        if self.transform is not None:
            frame_1 = self.transform(frame_1)
            frame_2 = self.transform(frame_2)
            frame_3 = self.transform(frame_3)

        return frame_1, frame_2, frame_3

# Example usage:
# Assuming that 'transform' is defined, and 'triplets_path' is the path to your dataset
# dataset = TripletDataset(triplets_path='path/to/triplet/folders', transform=transform)
# DataLoader can be used to create batches from the dataset


# import os
# from PIL import Image
# from torch.utils.data.dataset import Dataset


# class TripletDataset(Dataset):
#     """
#     Each triplet has its folder with the corresponding 3 images.
#     Structure of the triplet dataset:
#         triplet_<number>
#             -   1.png
#             -   2.png
#             -   3.png
#     """
#     def __init__(self, triplets_path, transform):
#         self.triplets_path = triplets_path
#         self.transform = transform
#         self.triplets = list(os.listdir(triplets_path))


#     def __len__(self):
#         return len(self.triplets)
    
#     def __getitem__(self, index):
#         current_path = self.triplets[index] # e.g. triplet triplet_000010
        
#         # Make sure you understand the __init__ function and the structure of the data first.

#         # TASK 1: Read the triplet and make sure you use the self.transform
#         frame_1 = ...
#         frame_2 = ...
#         frame_3 = ...

#         return frame_1, frame_2, frame_3
