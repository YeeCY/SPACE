from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class O2P2(Dataset):
    def __init__(self, root, mode):
        # checkpointdir = os.checkpointdir.join(root, mode)
        assert mode in ['train', 'test']
        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)

        self.img_paths = []
        img_dir = os.path.join(self.root, mode)
        for file in os.scandir(img_dir):
            img_path = file.path
            if 'png' in img_path or 'jpg' in img_path:
                self.img_paths.append(img_path)
        
        get_index = lambda x: int(os.path.basename(x).split('_')[0])
        self.img_paths.sort(key=get_index)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
        img = transform(img)

        return img
    
    def __len__(self):
        return len(self.img_paths)
    

