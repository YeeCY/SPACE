from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import ImageFile
from skimage import io

from fourrooms_env.envs.fourrooms import Fourrooms
from fourrooms_env.envs.fourrooms_multicoin import FourroomsMultiCoin


class FourroomsDataset(Dataset):
    def __init__(self, env):
        assert isinstance(env, (Fourrooms, FourroomsMultiCoin)), "'envs' must be instance of 'Fourrooms' or " \
                                                                 "'FourroomsMultiCoin'"

        try:
            num_state = env.observation_space.shape[0]
        except IndexError:
            num_state = env.observation_space.n

        # get observations
        import matplotlib
        matplotlib.use('TkAgg')
        observations = []
        for _ in range(20):
            for s in range(num_state):
                env.reset(s)
                obs = env.render(s)
                # io.imshow(obs)
                # io.show()
                # io.imsave("../images/{}.png".format(s), obs)
                observations.append(obs)
        self.observations = np.array(observations)

    def __getitem__(self, index):
        # img_path = self.img_paths[index]
        # img = io.imread(img_path)[:, :, :3]
        img = self.observations[index, :, :, :3]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
        img = transform(img)
        # io.imshow((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        # io.show()

        return img

    def __len__(self):
        return len(self.observations)
