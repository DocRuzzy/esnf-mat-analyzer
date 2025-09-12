import torch
from torch.utils.data import Dataset
import numpy as np
from esnf_mat_analyzer.validation.synthetic_ruler_generator import SyntheticRulerGenerator

class SyntheticRulerDataset(Dataset):
    """
    A PyTorch Dataset that generates synthetic ruler images on the fly.
    """

    def __init__(self, num_samples: int = 1000, width: int = 512, height: int = 512, transform=None):
        """
        Initializes the dataset.

        Args:
            num_samples: The total number of samples in the dataset.
            width: The width of the generated images.
            height: The height of the generated images.
            transform: Optional transform to be applied on a sample.
        """
        self.num_samples = num_samples
        self.width = width
        self.height = height
        self.transform = transform
        self.generator = SyntheticRulerGenerator(width, height)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generates a single sample of a synthetic ruler image and its label.
        """
        # Generate random properties for the ruler
        ruler_thickness = np.random.randint(5, 20)
        tick_width = np.random.randint(1, 3)
        tick_length = np.random.randint(10, 30)
        tick_spacing = np.random.randint(30, 100)
        rotation_angle = np.random.uniform(-30, 30)

        # Generate the ruler image
        image = self.generator.generate_ruler(
            ruler_thickness=ruler_thickness,
            tick_width=tick_width,
            tick_length=tick_length,
            tick_spacing=tick_spacing,
            rotation_angle=rotation_angle,
        )

        # For now, we generate random GP parameters as a placeholder.
        # In a real implementation, these would be calculated from the
        # ruler properties.
        gp_params = torch.randn(3)

        sample = {'image': image, 'gp_params': gp_params}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

if __name__ == '__main__':
    dataset = SyntheticRulerDataset(num_samples=10)
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample GP parameters: {sample['gp_params']}")
    print("SyntheticRulerDataset created successfully.")
