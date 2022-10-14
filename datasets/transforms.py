import random

import torch
from torchvision import transforms, utils
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import Compose, Lambda


class Transform(object):

    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __call__(self, add_jitter=False):
        if add_jitter:
            transform = transforms.Compose(
                [self.colorJitter(),
                 self.to_tensor(),
                 self.normalise()])
        else:
            transform = transforms.Compose(
                [self.to_tensor(), self.normalise()])
        return transform

    def to_tensor(self):
        return transforms.ToTensor()

    def normalise(self):
        return transforms.Normalize(self.mean.tolist(), self.std.tolist())

    def unnormalise(self):
        return transforms.Normalize((-self.mean / self.std).tolist(),
                                    (1.0 / self.std).tolist())

    def colorJitter(self):
        return transforms.ColorJitter(0.4, 0.2, 0.2, 0.1)


class ColorJitter(object):
    """
    Redefinition of ColorJitter class. This gives us more control over the
    jitter when running over a video sequence compared to torchvision which
    applies a different value for each sequence. We want to have consistent
    jitter across the video sequence.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.brightness_factor = None
        self.contrast_factor = None
        self.saturation_factor = None
        self.hue_factor = None

    def __call__(self, img):
        transform = self.get_params()
        return transform(img)

    def set_factors(self):
        assert self.hue < 0.5
        self.brightness_factor = random.uniform(1 - self.brightness,
                                                1 + self.brightness)
        self.contrast_factor = random.uniform(1 - self.contrast,
                                              1 + self.contrast)
        self.saturation_factor = random.uniform(1 - self.saturation,
                                                1 + self.saturation)
        self.hue_factor = random.uniform(-self.hue, self.hue)

    def get_params(self):
        transforms = []
        transforms.append(
            Lambda(
                lambda img: F.adjust_brightness(img, self.brightness_factor)))
        transforms.append(
            Lambda(lambda img: F.adjust_contrast(img, self.contrast_factor)))
        transforms.append(
            Lambda(
                lambda img: F.adjust_saturation(img, self.saturation_factor)))
        transforms.append(
            Lambda(lambda img: F.adjust_hue(img, self.hue_factor)))
        random.shuffle(transforms)
        transforms = Compose(transforms)
        return transforms
