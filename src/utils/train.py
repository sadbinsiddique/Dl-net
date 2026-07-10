from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class EmotionDataset(Dataset):
    """Custom Dataset for loading emotion images and labels."""
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):

        image = Image.open(
            self.df.loc[idx,"filepath"]
        ).convert("RGB")

        label = int(self.df.loc[idx,"class"])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transformations():
    """Returns a set of transformations for image preprocessing."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])