import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd

from skimage import io
import config

class FakeDetectionDataloader(data.Dataset):
# __init__ function is where the initial logic happens like reading a csv,
# assigning transforms etc.
    def __init__(self, csv_path):
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.img_path = np.asarray(self.data_info.iloc[1:, 0])
        # self.image_arr = np.asarray(self.data_info.iloc[1:, 1])
        # print(self.image_arr)
        # labels start from second column
        self.label_arr = np.asarray(self.data_info.iloc[1:, 1])

        # Calculate len
        self.data_len = len(self.data_info.index) - 1


    # __getitem__ function returns the data and labels. This function is
    # called from dataloader like this
    def __getitem__(self, index):
        # Get image name from the pandas df
        img_name = self.img_path[index]

        # Open image
        # print("image name:", img_name)
        img_as_img = Image.open(img_name)
        # print("before", np.array(img_as_img).shape)
        img_cropped = self.transform(img_as_img)
        # print("after",img_cropped.shape)

        # Get label(class) of the image based on the cropped pandas column
        # print(type(self.label_arr[index]))
        label = int(self.label_arr[index])
        # print(label, type(label))

        return (img_name, img_cropped, label)

    def __len__(self):
        return self.data_len

class FakeDetectionBCDataloader(data.Dataset):
# __init__ function is where the initial logic happens like reading a csv,
# assigning transforms etc.
    def __init__(self, csv_path):
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.img_path = np.asarray(self.data_info.iloc[1:, 0])
        self.xmin = np.asarray(self.data_info.iloc[1:, 1])
        self.ymin = np.asarray(self.data_info.iloc[1:, 2])
        self.xmax = np.asarray(self.data_info.iloc[1:, 3])
        self.ymax = np.asarray(self.data_info.iloc[1:, 4])
        # self.image_arr = np.asarray(self.data_info.iloc[1:, 1])
        # print(self.image_arr)
        # labels start from second column
        self.label_arr = np.asarray(self.data_info.iloc[1:, 5])

        # Calculate len
        self.data_len = len(self.data_info.index) - 1


    # __getitem__ function returns the data and labels. This function is
    # called from dataloader like this
    def __getitem__(self, index):
        # Get image name from the pandas df
        img_name = self.img_path[index]
        img_parts = self.img_path[index].split("_")
        img_load = '_'.join(img_parts[:-1]) + ".jpg"

        # Open image
        # print("image name:", img_name)
        # print("image load:", img_load)
        img_new = Image.open(img_load)
        crop_box = (self.xmin[index], self.ymin[index], self.xmax[index], self.ymax[index])
        crop_box = map(int, crop_box)
        img_as_img = img_new.crop((crop_box))
        # print("before", np.array(img_as_img).shape)
        img_cropped = self.transform(img_as_img)
        # print("after",img_cropped.shape)

        # Get label(class) of the image based on the cropped pandas column
        # print(type(self.label_arr[index]))
        label = int(self.label_arr[index])
        # print(label, type(label))

        return (img_name, img_cropped, label)

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    # Checking for pretraining dataloader
    # dataset = MyCustomDataset("annotatation_pretrain.csv")
    trainset = FakeDetectionBCDataloader("csv/test_B_LR_c0.csv")
    # valset = FakeDetectionDataloader("csv/val_HR_c0.csv")
    
    print("size of trainset, valset:", len(trainset))

    trainloader = torch.utils.data.DataLoader(trainset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=8
                            )

    # valloader = torch.utils.data.DataLoader(valset,
    #                             batch_size=config.batch_size,
    #                             shuffle=False,
    #                             num_workers=8
    #                             )

    batch = next(iter(trainloader))
    file_names, images, labels = batch
    print(images.shape, labels.shape)
    image = images[0].permute(1,2,0).detach().numpy() * 255.
    image = image.astype(np.uint8)      # Uint takes input range as 0-255
    print(image.shape)

    io.imsave("trial.png", image)
    print("Done")