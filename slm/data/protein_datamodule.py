from typing import *

import os
import random 
import pickle
from pathlib import Path
from glob import glob 
from functools import lru_cache
import tree

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from lightning import LightningDataModule
from esm.utils.constants import esm3 as C


def random_truncate(data_dict, max_len=512, non_moving_ids: Optional[List] = None):
    # 1000 be the max length of dataset
    L = len(data_dict['sequence'])
    if max_len and max_len > 0 and L > max_len:
        # Randomly truncate
        max_idx = L - max_len
        cropped_idx = []
        if non_moving_ids is not None:  # centered cropping
            # start and end should randomly include the min-max of non-moving ids
            start = np.random.randint(0, min(max_idx + 1, max(1, min(non_moving_ids))))
        else: # fully random
            start = np.random.randint(0, max_idx + 1)
        end = start + max_len
        data_dict = tree.map_structure(
            lambda x: x[start : end ], data_dict)
    return data_dict


class ESMEmbeddingDataset(Dataset):
    """Random access to pickle protein objects of dataset.
    """
    def __init__(
        self, 
        path_to_embeddings: Optional[str] = None,
        cluster_rep_csv: Optional[str] = None,
        transform: Optional[Callable] = None, 
        training: bool = True,
        max_len: int = -1,
        **kwargs,
    ):
        super().__init__()
        
        # Pre-computed embeddings. (eg., ESM2)
        self.path_to_embeddings = Path(path_to_embeddings)
        assert self.path_to_embeddings.is_dir(), f"Invalid directory: {self.path_to_embeddings}"
        self.data = list(self.path_to_embeddings.iterdir())
        if cluster_rep_csv:
            cluster_rep_seq = set(pd.read_csv(cluster_rep_csv)['rep_seq'].to_list())
            print(f">>> Filtering targets = {len(self.data)} files")
            self.data = [p for p in self.data if p.stem in cluster_rep_seq]
        
        if kwargs.get('dev', False):
            self.data = self.data[:500]
        
        assert len(self.data) > 1, f"Empty directory: {self.data}"
        print(f">>> Loaded {len(self.data)} embedding files")
        
        self.transform = transform
        self.training = training  # not implemented yet
        self.max_len = max_len
        
    @property
    def num_samples(self):
        return len(self.data)
    
    def len(self): 
        return self.__len__()

    def __len__(self):
        return self.num_samples 

    def get(self, idx):
        return self.__getitem__(idx)
    
    @property
    def accession_codes(self):
        return [data_path.stem for data_path in self.data]
    
    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        """direct loading
        """
        data_path = self.data[idx]
        accession_code = self.accession_codes[idx]
        _data_dict = torch.load(data_path, map_location='cpu')
        # strip tensors
        data_dict = {}
        for k, v in _data_dict.items():
            data_dict[k] = v[1:-1] if torch.is_tensor(v) and k != "coordinates" else v
            assert len(data_dict[k]) == len(_data_dict['sequence']), f"{k} {len(data_dict[k])} != {len(_data_dict['sequence'])}"
            if "tokens" in k:
                data_dict[k] = data_dict[k].to(torch.long)

        data_dict["mask"] = torch.ones_like(data_dict["structure_tokens"])
        # Apply data transform.
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        if self.training:
            data_dict = random_truncate(data_dict, self.max_len)

        data_dict['accession_code'] = accession_code
        return data_dict  # dict of tensors


class BatchTensorConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """
    def __init__(self, target_keys: Optional[List] = None):
        self.target_keys = target_keys
    
    def __call__(self, raw_batch: Sequence[Dict[str, object]]):
        B = len(raw_batch)
        # Only do for Tensor
        target_keys = self.target_keys \
            if self.target_keys is not None else [k for k,v in raw_batch[0].items() if torch.is_tensor(v)]

        # Non-array, for example string, int
        non_tensor_keys = [k for k in raw_batch[0] if k not in target_keys]
        collated_batch = dict()
        
        for k in target_keys:
            if k == 'structure_tokens':
                pad_v = C.STRUCTURE_PAD_TOKEN
            elif k == "sequence_tokens":
                pad_v = C.SEQUENCE_PAD_TOKEN
            else:
                pad_v = 0
            collated_batch[k] = self.collate_dense_tensors([d[k] for d in raw_batch], pad_v=pad_v)
        
        for k in non_tensor_keys:    # return non-array keys as is
            collated_batch[k] = [d[k] for d in raw_batch]
        return collated_batch

    @staticmethod
    def collate_dense_tensors(samples: Sequence, pad_v: float = 0.0):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result



class ProteinDataModule(LightningDataModule):
    """`LightningDataModule`. Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: Optional[torch.utils.data.Dataset] = None,
        dataset_train: Optional[torch.utils.data.Dataset] = None,
        dataset_val: Optional[torch.utils.data.Dataset] = None,
        dataset_test: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 64,
        generator_seed: int = 42,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
    ) -> None:
        """
        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.dataset: Optional[Dataset] = dataset
        self.data_train: Optional[Dataset] = dataset_train
        self.data_val: Optional[Dataset] = dataset_val
        self.data_test: Optional[Dataset] = dataset_test

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if stage == 'fit':
            if not self.data_train and not self.data_val:
                self.data_train, self.data_val = random_split(
                    dataset=self.dataset,
                    lengths=self.hparams.train_val_split,
                    generator=torch.Generator().manual_seed(self.hparams.generator_seed),
                )
                # self.data_val.dataset.training = False
            else:
                assert self.data_train and self.data_val, "Split datasets using defined configurations."
        elif stage in ('predict', 'test') and not self.data_test:
            self.data_test = self.dataset
            self.data_test.training = False
        else:
            raise NotImplementedError(f"Stage {stage} not implemented.")
        
    def _dataloader_template(self, dataset: Dataset[Any], training=True) -> DataLoader[Any]:
        """Create a dataloader from a dataset.

        :param dataset: The dataset.
        :return: The dataloader.
        """
        batch_collator = BatchTensorConverter()    # list of dicts -> dict of tensors
        return DataLoader(
            dataset=dataset,
            collate_fn=batch_collator,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle and training,
        )
    
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._dataloader_template(self.data_train)
           

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self._dataloader_template(self.data_val, training=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._dataloader_template(self.data_test, training=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

