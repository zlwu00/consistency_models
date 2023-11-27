import copy
import math
import os
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
from torch import nn
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.utils.torch_utils import randn_tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn, optim
from torchmetrics import MeanMetric
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image

from consistency.pipeline import ConsistencyPipeline

with suppress(ImportError):
    import wandb


class Consistency(LightningModule):
    def __init__(
        self,
        model: UNet2DModel,
        *,
        loss_fn: nn.Module = nn.MSELoss(),
        learning_rate: float = 1e-4,
        data_std: float = 0.5,
        time_min: float = 0.002,
        time_max: float = 80.0,
        bins_min: int = 2,
        bins_max: int = 150,
        bins_rho: float = 7,
        initial_ema_decay: float = 0.9,
        optimizer_type: Type[optim.Optimizer] = optim.RAdam,
        samples_path: str = "samples/",
        save_samples_every_n_epoch: int = 10,
        num_samples: int = 16,
        sample_steps: int = 1,
        use_ema: bool = True,
        sample_seed: int = 0,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        push_every_n_steps: Optional[int] = 100,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = model
        self.model_ema = copy.deepcopy(model)
        self.image_size = model.sample_size
        self.channels = model.in_channels

        self.model_ema.requires_grad_(False)

        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate
        self.initial_ema_decay = initial_ema_decay

        self.data_std = data_std
        self.time_min = time_min
        self.time_max = time_max
        self.bins_min = bins_min
        self.bins_max = bins_max
        self.bins_rho = bins_rho

        self._loss_tracker = MeanMetric()
        self._bins_tracker = MeanMetric()
        self._ema_decay_tracker = MeanMetric()

        Path(samples_path).mkdir(exist_ok=True, parents=True)

        self.samples_path = samples_path
        self.save_samples_every_n_epoch = save_samples_every_n_epoch
        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.use_ema = use_ema
        self.sample_seed = sample_seed

        self.MLP = nn.Linear(32, 32 * 32 * 3)
        self.BatchNorm = nn.BatchNorm1d(32 * 32 * 3)
        self.hidden_size = 32

        self.push_to_hub = True if model_id else False

        if self.push_to_hub:
            self.token = token
            self.model_id = model_id
            self.push_every_n_steps = push_every_n_steps
            self.repo = None

    @rank_zero_only
    def setup(self, *args, **kwargs) -> None:
        '''
        Connect to huggingface and push the trained model up to own space
        '''
        if self.push_to_hub:
            import huggingface_hub

            if self.token is None:
                self.token = huggingface_hub.HfFolder.get_token()

            username = huggingface_hub.whoami(self.token)["name"]
            full_repo_name = f"{username}/{self.model_id}"

            huggingface_hub.create_repo(full_repo_name, exist_ok=True, token=self.token)

            self.repo = huggingface_hub.Repository(
                local_dir=self.model_id,
                clone_from=full_repo_name,
                token=self.token,
            )

            with open(os.path.join(self.model_id, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")

    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
    ):
        return self._forward(self.model, images, times)

    def _forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        times: torch.Tensor,
        clip: bool = True,
    ):
    #
        skip_coef = self.data_std**2 / (
            (times - self.time_min).pow(2) + self.data_std**2
        )
        out_coef = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)

        out: UNet2DOutput = model(images, times)

        out = self.image_time_product(
            images,
            skip_coef,
        ) + self.image_time_product(
            out.sample,
            out_coef,
        )

        if clip:
            return out.clamp(-1.0, 1.0)

        return out

    def training_step(self, images: torch.Tensor, *args, **kwargs):
        _bins = self.bins
        
        noise = images[1]
        noise = self.MLP(noise)
        noise = self.BatchNorm(noise)
        
        batch_size, channel, h, w = images[0].shape
        
        noise = noise.reshape(batch_size, channel, h, w)

        # noise_disturbation = torch.randn(images[0].shape, device=images[0].device) * 0.3

        # noise = noise + noise_disturbation

        timesteps = torch.randint(
            0,
            _bins - 1,
            (images[0].shape[0],),
            device=images[0].device,
        ).long()


        current_time = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps + 1, _bins)

        current_noise_image = images[0] + self.image_time_product(
            noise,
            current_time
        )

        next_noise_image = images[0] + self.image_time_product(
            noise,
            next_times,
        )

        with torch.no_grad():
            target1 = self._forward(
                self.model_ema,
                current_noise_image,
                current_time,
            )

        #target2 = images[0].clamp(-1.0, 1.0)
        

        loss = self.loss_fn(self(next_noise_image, next_times), target1)
        #loss2 = self.loss_fn(self(next_noise_image, next_times), target2)


        self._loss_tracker(loss)
        self.log(
            "loss",
            self._loss_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        self._bins_tracker(_bins)
        self.log(
            "bins",
            self._bins_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), lr=self.learning_rate)

    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs):
        if (
            self.push_to_hub
            and self.trainer.global_step % self.push_every_n_steps == 0
            and self.trainer.global_step > 0
        ):
            pipeline = ConsistencyPipeline(
                unet=self.model_ema if self.use_ema else self.model, 
                MLP=self.MLP, 
                norm=self.BatchNorm,
            )

            pipeline.save_pretrained(self.model_id)

            self.repo.push_to_hub(
                commit_message=f"Step {self.global_step}",
                blocking=False,
            )

            del pipeline

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_update()

    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

        self._ema_decay_tracker(self.ema_decay)
        self.log(
            "ema_decay",
            self._ema_decay_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    @property
    def ema_decay(self):
        return math.exp(self.bins_min * math.log(self.initial_ema_decay) / self.bins)

    @property
    def bins(self) -> int:
        return math.ceil(
            math.sqrt(
                self.trainer.global_step
                / self.trainer.estimated_stepping_batches
                * (self.bins_max**2 - self.bins_min**2)
                + self.bins_min**2
            )
        )

    def timesteps_to_times(self, timesteps: torch.LongTensor, bins: int):
        return (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )

    @rank_zero_only
    def on_train_start(self) -> None:
        self.save_samples(
            f"{0:05}",
            num_samples=self.num_samples,
            steps=self.sample_steps,
            generator=torch.Generator(device=self.device).manual_seed(self.sample_seed),
            use_ema=self.use_ema,
        )

    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        if (
            (self.trainer.current_epoch < 30)
            or ((self.trainer.current_epoch + 1) % self.save_samples_every_n_epoch == 0)
            or self.trainer.current_epoch == (self.trainer.max_epochs - 1)
        ):
            self.save_samples(
                f"{(self.current_epoch+1):05}",
                num_samples=self.num_samples,
                steps=self.sample_steps,
                generator=torch.Generator(device=self.device).manual_seed(
                    self.sample_seed
                ),
                use_ema=self.use_ema,
            )

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        steps: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        shape = (num_samples,self.hidden_size)
        noise = randn_tensor(shape, generator=generator, device=self.device)
        noise = self.MLP(noise)
        noise = self.BatchNorm(noise)
        
        noise = noise.reshape(num_samples, self.channels, self.image_size, self.image_size)

        # shape = (num_samples, self.channels, self.image_size, self.image_size) 

        time = torch.tensor([self.time_max], device=self.device)

        images: torch.Tensor = self._forward(
            self.model_ema if use_ema else self.model,
            #randn_tensor(shape, generator=generator, device=self.device) * time,
            noise * time,
            time,
        )

        if steps <= 1:
            return images

        _timesteps = list(
            reversed(range(0, self.bins_max, self.bins_max // steps - 1))
        )[1:]
        _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]

        times = self.timesteps_to_times(
            torch.tensor(_timesteps, device=self.device), bins=150
        )

        for time in times:
            noise = randn_tensor(shape, generator=generator, device=self.device)
            noise = self.MLP(noise)
            noise = self.BatchNorm(noise)
            noise = noise.reshape(num_samples, self.channels, self.image_size, self.image_size)
            # noise = randn_tensor(shape, generator=generator, device=self.device)
            images = images + math.sqrt(time.item() ** 2 - self.time_min**2) * noise
            images = self._forward(
                self.model_ema if use_ema else self.model,
                images,
                time[None],
            )

        return images

    @torch.no_grad()
    def save_samples(
        self,
        filename: str,
        num_samples: int = 16,
        steps: int = 1,
        use_ema: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        samples = self.sample(
            num_samples=num_samples,
            steps=steps,
            generator=generator,
            use_ema=use_ema,
        )
        samples.mul_(0.5).add_(0.5)
        grid = make_grid(
            samples,
            nrow=math.ceil(math.sqrt(samples.size(0))),
            padding=self.image_size // 16,
        )

        save_image(
            grid,
            f"{self.samples_path}/{filename}.png",
            "png",
        )

        if isinstance(self.trainer.logger, WandbLogger):
            wandb.log(
                {
                    "samples": wandb.Image(to_pil_image(grid)),
                },
                commit=False,
                step=self.trainer.global_step,
            )

        del samples
        del grid
        torch.cuda.empty_cache()

    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b c h w, b -> b c h w", images, times)
    
   
