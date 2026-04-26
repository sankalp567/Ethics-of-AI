import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import Config

class CelebADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and csv files.
            split (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'imgs')
        self.transform = transform
        
        # Load attributes and partitions
        attr_file = os.path.join(root_dir, 'list_attr_celeba.csv')
        partition_file = os.path.join(root_dir, 'list_eval_partition.csv')
        
        # The attr file has index 'image_id'
        self.df_attr = pd.read_csv(attr_file)
        self.df_partition = pd.read_csv(partition_file)
        
        # Merge them
        self.df = pd.merge(self.df_attr, self.df_partition, on='image_id')
        
        # Replace -1 with 0 for binary labels
        self.df = self.df.replace(-1, 0)
        
        # Filter by partition
        partition_map = {'train': 0, 'val': 1, 'test': 2}
        self.df = self.df[self.df['partition'] == partition_map[split]].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['image_id'])
        image = Image.open(img_name).convert('RGB')

        target = self.df.iloc[idx][Config.TARGET_ATTR]

        # return multiple sensitives
        sensitives = {
            attr: int(self.df.iloc[idx][attr])
            for attr in Config.SENSITIVE_ATTRS
        }

        if self.transform:
            image = self.transform(image)

        return image, target, sensitives
    
    
def get_celeba_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])),
        transforms.ToTensor(),
    ])

    train_dataset = CelebADataset(root_dir=Config.DATA_DIR, split='train', transform=train_transform)
    val_dataset = CelebADataset(root_dir=Config.DATA_DIR, split='val', transform=val_test_transform)
    test_dataset = CelebADataset(root_dir=Config.DATA_DIR, split='test', transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    return train_loader, val_loader, test_loader
