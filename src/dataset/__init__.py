from .atari import Atari
from .obj3d import Obj3D
from .fourrooms import Fourrooms, FourroomsMultiCoin, FourroomsDataset
from .o2p2 import O2P2
from torch.utils.data import DataLoader


__all__ = ['get_dataset', 'get_dataloader']

def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if cfg.dataset == 'ATARI':
        mode = 'validation' if mode == 'val' else mode
        return Atari(cfg.dataset_roots.ATARI, mode, gamelist=cfg.gamelist)
    elif cfg.dataset == 'OBJ3D_SMALL':
        return Obj3D(cfg.dataset_roots.OBJ3D_SMALL, mode)
    elif cfg.dataset == 'OBJ3D_LARGE':
        return Obj3D(cfg.dataset_roots.OBJ3D_LARGE, mode)
    elif cfg.dataset == 'FOURROOMS_MULTICOIN':
        env = FourroomsMultiCoin(random_coin=True)
        return FourroomsDataset(env)
    elif cfg.dataset == 'O2P2':
        return O2P2(cfg.dataset_roots.O2P2, mode)
    else:
        raise TypeError("Invalid dataset!")

def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']
    
    batch_size = getattr(cfg, mode).batch_size
    shuffle = True if mode == 'train' else False
    num_workers = getattr(cfg, mode).num_workers
    
    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

