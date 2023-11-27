import torch
import torch.nn as nn
import json, os

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from diffusers import UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from consistency import Consistency
from consistency.loss import PerceptualLoss

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from vit import ViTForClassfication


DATASET_NAME = "cifar10"
RESOLUTION = 32
BATCH_SIZE = 128
MAX_EPOCHS = 200
LEARNING_RATE = 1e-5
MODEL_ID = f"cm-{DATASET_NAME}-{RESOLUTION}-fix_noise"

SAMPLES_PATH = "./samples"
NUM_SAMPLES = 64
SAMPLE_STEPS = 1  # Set this value larger if you want higher sample quality.






class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, dataset_config_name=None):
        self.dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split="train",
        )
        self.noise_dimension = 32
        self.image_key = [
            key for key in ("image", "img") if key in self.dataset[0]
        ][0]
        self.augmentations = transforms.Compose(
        [         
            transforms.Resize(
                                RESOLUTION,
                                interpolation=transforms.InterpolationMode.BILINEAR,
                                ),
            transforms.CenterCrop(RESOLUTION),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        )
        # self.dir = "/data/local2/zw2599/repos/consistency-models/experiments/vit-with-100-epochs"
        # self.configfile = os.path.join(self.dir, 'config.json')
        # with open(self.configfile, 'r') as f:
        #     self.config = json.load(f)
   
        # self.vit_model = ViTForClassfication(self.config)
        # self.cpfile = os.path.join(self.dir, "model_final.pt")
        # self.vit_model.load_state_dict(torch.load(self.cpfile))
        # self.vit_model.classifier = Identity()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        original_image = self.augmentations(self.dataset[index][self.image_key].convert("RGB"))

        torch.manual_seed(index)

        noise = torch.randn(self.noise_dimension)

     

        # self.vit_model.eval()

        # with torch.no_grad():
        #   noise_image = self.vit_model(original_image.unsqueeze(0))
        #   noise_image = noise_image[0].squeeze()
        
        return original_image, noise

dataloader = DataLoader(
    Dataset(DATASET_NAME),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)


consistency = Consistency(
    model=UNet2DModel(
        sample_size=RESOLUTION,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(128, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    ),
    # You could use multiple net types. 
    # Recommended setting is "squeeze" + "vgg"
    # loss_fn=PerceptualLoss(net_type=("squeeze", "vgg"))
    # See https://github.com/richzhang/PerceptualSimilarity
    #loss_fn=PerceptualLoss(net_type="squeeze"), 
    learning_rate=LEARNING_RATE,
    samples_path=SAMPLES_PATH,
    save_samples_every_n_epoch=1,
    num_samples=NUM_SAMPLES,
    sample_steps=SAMPLE_STEPS,
    use_ema=True,
    sample_seed=42,
    model_id=MODEL_ID,
)

trainer = Trainer(
    accelerator="cuda",
    logger=WandbLogger(project="consistency", log_model=True),
    devices=1,
    callbacks=[
        ModelCheckpoint(
            dirpath="ckpt", 
            save_top_k=3, 
            monitor="loss",
        )
    ],
    max_epochs=MAX_EPOCHS,
    precision=16 if torch.cuda.is_available() else 32,
    log_every_n_steps=50,
    gradient_clip_algorithm="norm",
    gradient_clip_val=1.0,
)

trainer.fit(consistency, dataloader)
