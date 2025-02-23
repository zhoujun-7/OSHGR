import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data.distributed import DistributedSampler


class BaseTrainer:
    def __init__(self):
        self.DTYPE = torch.float32
        self.IS_GRAD_CLIP = True
        self.IS_COMPILE = False
        self.GRAD_ACCM = 2
        self.WORLD_SIZE = 1
        self.EPOCH_START = 0
        self.EPOCH_END = 200
        self.SEED = 1
        self.LR = 0.0001
        self.PER_GPU_BATCH = 8
        self.MAX_GRAD_NORM = 1.0

    def setup_seed(self, seed):
        if seed == 0:
            torch.backends.cudnn.benchmark = True
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setup_ddp_env(self, rank, port="29500"):
        if self.WORLD_SIZE > 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = port
            distributed.init_process_group(
                "nccl",
                rank=rank,
                world_size=self.WORLD_SIZE,
            )

    def setup_model(self):
        self.model = nn.Linear(1000, 3000, bias=False)
        self.model.to(self.RANK)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.ddp_model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.RANK],
            output_device=self.RANK,
        )
        if self.IS_COMPILE:
            self.ddp_model = torch.compile(self.ddp_model)
        distributed.barrier()

    def setup_ddp_dataset(self):
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                pass

            def __len__(self):
                return 1000

            def __getitem__(self, index):
                data = torch.randn(1000)
                return index, data

        dataset = DummyDataset()
        self.sampler = DistributedSampler(dataset)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.PER_GPU_BATCH,
            sampler=self.sampler,
            drop_last=True,
        )

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=self.LR)
        self.scaler = GradScaler()  # for half-precision training.

    def save_ckpt(self, save_path):
        if self.RANK == 0:
            torch.save(self.ddp_model.module.state_dict(), save_path)

    def clean(self):
        if self.WORLD_SIZE > 1:
            distributed.destroy_process_group()
            print(f"Rank {self.RANK} is done.")

    def ddp_worker(self, rank):
        self.RANK = rank
        self.DEVICE = rank
        self.setup_seed(self.SEED + rank)
        self.setup_ddp_env(rank)
        self.setup_ddp_dataset()
        self.setup_model()
        self.setup_optimizer()

        time_start = time.time()
        for epoch in range(self.EPOCH_START, self.EPOCH_END):
            self.sampler.set_epoch(epoch)
            for i, (index, data) in enumerate(self.dataloader):
                with torch.autocast(device_type="cuda", dtype=self.DTYPE):
                    output = self.ddp_model(data)
                    loss = (output**2).mean()

                self.scaler.scale(loss).backward()
                if i % self.GRAD_ACCM == 0:
                    if self.IS_GRAD_CLIP:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.ddp_model.parameters(),
                            max_norm=self.MAX_GRAD_NORM,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                torch.cuda.synchronize(device=rank)

                if self.RANK == 0 and i == 0:
                    info = f"epoch: {epoch} \t loss:{loss:.2e} \t index:{index[0]}"
                    print(info)
        time_end = time.time()
        print(f"duration time: {time_end-time_start:.2f}s")
        self.save_ckpt("./tmp.ckpt")
        self.clean()

    def ddp_run(self, fn, world_size):
        if world_size > 1:
            mp.spawn(fn, nprocs=world_size, join=True)
        else:
            fn()


if __name__ == "__main__":
    trainer = BaseTrainer()
    trainer.WORLD_SIZE = 1
    trainer.ddp_run(
        trainer.ddp_worker,
        trainer.WORLD_SIZE,
    )
