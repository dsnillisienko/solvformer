import dataclasses

import torch

from solvformer.model import Config, Solvformer


def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_checkpoint(path, model, **extra):
    """Конфиг архитектуры лежит рядом с весами, так что по одному чекпойнту модель собирается заново."""
    torch.save({'cfg': dataclasses.asdict(model.cfg), 'model': model.state_dict(), **extra}, path)


def read_checkpoint(path):
    """В старых чекпойнтах оценка на валидации сохранена numpy-скаляром, а weights_only=True такие не читает,
    поэтому чекпойнт читается полным unpickle. Загружать стоит только чекпойнты из доверенного источника."""
    return torch.load(path, map_location='cpu', weights_only=False)


def load_model(path, device):
    ck = read_checkpoint(path)
    model = Solvformer(Config(**ck['cfg']))
    model.load_state_dict(ck['model'])
    return model.to(device).eval()
