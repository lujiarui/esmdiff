import os
from typing import Any, Dict, Tuple, Optional, Union
from random import random
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from esm.utils.constants import esm3 as C

from slm.utils.noise_utils import get_inputs_for_mlm
from slm.models.net import TimestepEmbedder
from slm.models.utils import cross_entropy
from slm.utils import noise_utils


def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(
        * x.shape,
        * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class EncoderOutput:
    """Class for keeping track of an item in inventory."""
    last_hidden_state: torch.Tensor


class LanguageModeling(LightningModule):
    """Example of a `LightningModule` for language modeling training.
    
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        # network and diffusion module
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.compile = compile
    
    def model_step(self, batch, training=True):
        target = batch["structure_tokens"]  # (B, L)
        mask = batch["mask"] * (target != C.STRUCTURE_PAD_TOKEN)
        target[target == C.STRUCTURE_PAD_TOKEN] = -100

        # https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/t5/modeling_t5.py#L782
        hiddens = torch.zeros_like(batch["embeddings"])
        feat_dict = {
            "labels": target,
            "encoder_outputs": [hiddens, None, None], # (B, L, D)
            "attention_mask": mask,
            "return_dict": True,
        }
        outs = self.net(**feat_dict)
        logits = outs["logits"]
        loss = outs["loss"].mean()
        
        pred = torch.argmax(logits, dim=-1) # [B, L, V] -> [B, L]
        acc = ((pred == target) * mask).sum() / mask.sum()
        
        # Breakdown of each loss
        loss_bd = {
            "nll": loss.detach().clone(),
            "acc": acc.detach().clone(),
        }
        return loss, loss_bd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("")

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_bd = self.model_step(batch)
        
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for k,v in loss_bd.items():
            if k == 'loss': continue    # log loss separately for epoch
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_bd = self.model_step(batch, training=False)

        # update and log metrics
        self.val_loss(loss) # update
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k,v in loss_bd.items():
            if k == 'loss': continue    # log loss separately for epoch
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        _vall = self.val_loss.compute()  # get current val acc
        self.val_loss_best(_vall)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        raise NotImplementedError("Test step not implemented.")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass
    
    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> str:
        n_samples = 100
        assert len(batch["embeddings"]) == 1, f"Batch size must be 1, but got {len(batch['embeddings'])}."
        hiddens = batch["embeddings"].repeat(n_samples, 1, 1)
        # make dataclass
        enc_out = EncoderOutput(last_hidden_state=hiddens)

        feat_dict = {
            # "inputs": hiddens, # the method initializes it with bos_token_id and a batch size of 1
            "encoder_outputs": enc_out,
            "max_length": hiddens.size(1),
            "min_length": hiddens.size(1),
            "do_sample": True,
            "num_beams": 1,
            "temperature": 1.0,
        }
        # https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate
        outputs = self.net.generate(**feat_dict)
        print("outputs", outputs.shape)
        return outputs # longtensors
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "name": "trainer/lr",

                    # "frequency": 1000,
                    # "monitor": "val/loss",
                    # "interval": "epoch",
                    # "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class JointLanguageModeling(LanguageModeling):
    def __init__(self, *args, **kwargs):
        """Language modeling for joint distribution of structure and sequence tokens.
        The input will be concatenation of both sequence and structure tokens.
        """
        super().__init__(*args, **kwargs)

    def model_step(self, batch, training=True):
        target_str = batch["structure_tokens"].detach().clone()  # (B, L)
        target_seq = batch["sequence_tokens"].detach().clone()
        mask = torch.nan_to_num(batch["mask"]) * (target_str != C.STRUCTURE_PAD_TOKEN)

        # https://github.com/evolutionaryscale/esm/blob/main/esm/utils/constants/esm3.py
        # for CE loss, -100 is ignored
        target_str[target_str == C.STRUCTURE_PAD_TOKEN] = -100        
        target_seq[target_seq == C.SEQUENCE_PAD_TOKEN] = -100
        # concat sequence and structure tokens
        target = torch.cat([target_seq, target_str], dim=1)
        
        feat_dict = {
            "labels": target,  # -100 is ignored    # only for training, (B, 2*L)
            "sequence_embeddings": batch["embeddings"], # (B, L, D)
            "structure_tokens": batch["structure_tokens"],
            "mask": mask,   # singleton mask
        }

        # print("shapes", {k:v.shape for k,v in feat_dict.items()})
        
        outs = self.net(**feat_dict)
        loss = outs["loss"].mean()

        # Breakdown of each loss
        loss_bd = {
            "nll": loss.detach().clone(),
            "sequence_nll": outs["sequence_nll"].mean().detach().clone(),
            "structure_nll": outs["structure_nll"].mean().detach().clone(),
            "sequence_acc": outs["sequence_acc"].mean().detach().clone(),
            "structure_acc": outs["structure_acc"].mean().detach().clone(),
        }
        return loss, loss_bd


class ConditionalLanguageModeling(LanguageModeling):
    def model_step(self, batch, training=True):
        target = batch["structure_tokens"]  # (B, L)
        mask = batch["mask"] * (target != C.STRUCTURE_PAD_TOKEN)
        target[target == C.STRUCTURE_PAD_TOKEN] = -100
        
        feat_dict = {
            "labels": target,
            "inputs_embeds": batch["embeddings"],   # (B, L, D)
            "attention_mask": mask,
            "return_dict": True,
        }
        outs = self.net(**feat_dict)
        logits = outs["logits"]
        loss = outs["loss"].mean()
        
        pred = torch.argmax(logits, dim=-1) # [B, L, V] -> [B, L]
        acc = ((pred == target) * mask).sum() / mask.sum()
        
        # Breakdown of each loss
        loss_bd = {
            "nll": loss.detach().clone(),
            "acc": acc.detach().clone(),
        }
        return loss, loss_bd


class MaskedDiffusionLanguageModeling(LanguageModeling):
    """Conditional Masked Diffusion Language Model for protein structure tokens.
    
    Inspired by https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py.
    """
    def __init__(
        self,
        # modules
        noise_schedule: noise_utils.Noise = None,
        sigma_embedder: nn.Module = None,
        # main flags
        time_conditioning: bool = False,
        change_of_variables: bool = False,
        importance_sampling: bool = False,
        condition_dropout: float = 0.0,
        condition_mask_rate: float = 0.5,
        sequence_prediction: bool = False,
        # not change this
        T: int = 0,
        sampling_eps: float = 1e-3,
        # sampler
        antithetic_sampling: bool = True,
        noise_removal: bool = True,
        structure_only: bool = False,
        coupled_condition_mask: bool = False,
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if noise_schedule is None: 
            print("Using default noise schedule: CosineNoise(eps=1e-3)")
            noise_schedule = noise_utils.CosineNoise(eps=1e-3)
        
        self.noise = noise_schedule    # nn.module
        self.sigma_embedder = sigma_embedder  # nn.module (B) -> (B, D)

        # flags
        self.antithetic_sampling = antithetic_sampling
        self.change_of_variables = change_of_variables
        self.importance_sampling = importance_sampling
        self.time_conditioning = time_conditioning
        self.T = T  # for discrete time markov chain
        self.sampling_eps = sampling_eps
        self.noise_removal = noise_removal
        self.structure_only = structure_only
        self.coupled_condition_mask = coupled_condition_mask
        
        # new features
        self.condition_dropout = condition_dropout
        self.condition_mask_rate = condition_mask_rate
        self.sequence_prediction = sequence_prediction
        
        # check
        if condition_dropout:
            assert 0.0 <= condition_dropout < 1.0, f"Invalid condition_dropout: {condition_dropout}"
        if condition_mask_rate:
            assert not coupled_condition_mask, f"coupled_condition_mask has to be False when condition_mask_rate is not 0.0"
            assert 0.0 <= condition_mask_rate < 1.0, f"Invalid condition_mask_rate: {condition_mask_rate}"
        if sequence_prediction:
            assert self.net.output_heads.sequence_head is not None, f"Sequence head not found in tbe network, but sequence_prediction is True."

        assert not (self.change_of_variables and self.importance_sampling)

        # constants
        self.vocab_size = 4101  # VQVAE_CODEBOOK_SIZE + 5 special tokens
        self.mask_index = C.STRUCTURE_MASK_TOKEN
        self.condition_mask_index = C.SEQUENCE_MASK_TOKEN
        self.neg_infinity = -1000000.0

    # for finetuning "esm3" model (BERT)
    def model_step(self, batch, training=True):
        labels = batch["structure_tokens"].detach().clone()
        x0 = batch["structure_tokens"].detach().clone()
        condition_seq = batch["sequence_tokens"]     # (B, L)
        B, L = labels.shape
        
        if self.condition_dropout > 0 and training: # in face only training/Validation call this model_step() in this class
            if random() < self.condition_dropout:
                condition_seq = None

        if self.condition_mask_rate > 0 and condition_seq is not None and training:
            mask = (torch.rand_like(condition_seq, dtype=torch.float) < self.condition_mask_rate) & (condition_seq != C.SEQUENCE_PAD_TOKEN)
            condition_seq = torch.where(mask, C.SEQUENCE_MASK_TOKEN, condition_seq) # mutate valid tokens to [MASK](seq)
        
        # Padding (B, L)
        loss_mask = batch["mask"] * (labels != C.STRUCTURE_PAD_TOKEN)
        
        # diffusion masking, get xt and time
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:  # round time step
            t = (t * self.T).to(torch.int) / self.T # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)

        if self.change_of_variables:
            net_conditioning = t[:, None]
            f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
            f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            net_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        if self.structure_only: # no condition_seq
            condition_seq = None 

        xt, condition_seq = self.q_xt(x0, move_chance, condition_seq=condition_seq, non_moving_mask=batch.get("non_moving_mask", None))
        ############################
        # forward pass
        logits, seq_logits = self._model_wrapper(xt, condition_seq, net_conditioning)
        ############################
        if torch.isnan(logits).any():
            print("Model output logits", logits)
        
        # score parameterization, continuous time.
        log_p_theta = torch.gather(
            input=logits,
            dim=-1,
            index=x0[:, :, None],
        ).squeeze(-1)
        
        if self.change_of_variables or self.importance_sampling:
            loss = log_p_theta * torch.log1p(- torch.exp(- self.noise.sigma_min))
        else:
            # dsigma: (B, )
            # NLL loss
            loss = - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
        
        loss = (loss * loss_mask).sum() / loss_mask.sum()   # fl
        loss_bd = {"nelbo": loss.detach().clone()}
        
        # auxilliary loss: always recover sequence
        if self.sequence_prediction:
            assert seq_logits is not None, f"Sequence logits not found in the forward pass."
            # (B, L, V)
            seq_reconstruction = cross_entropy(
                seq_logits, batch["sequence_tokens"], ignore_index=C.SEQUENCE_PAD_TOKEN
            )
            # note that here the loss mask is equivalent to non-padding mask
            # which different from "only count the masked tokens"
            # so it is readily applicable for auxilliary loss
            seq_reconstruction = (seq_reconstruction * loss_mask).sum() / loss_mask.sum()
            loss = loss + seq_reconstruction
            loss_bd["seq_nll"] = seq_reconstruction.detach().clone()

        return loss, loss_bd
    
    def _model_wrapper(self, xt, sequence_tokens=None, sigma=None, shield_special_tokens=False):
        # create time condition
        if sigma is not None:
            model_dtype = self.sigma_embedder.parameters().__next__().dtype
            sigma = self._process_sigma(sigma) # align with xt
            sigma = sigma.to(model_dtype)
            conditions = self.sigma_embedder(sigma)
            conditions = torch.tile(conditions[:, None, :], (1, xt.shape[1], 1))
        else:
            conditions = None   # no time conditioning (vanilla finetuning of BERT)
        
        _forward_output = self.net(
            structure_tokens=xt,
            sequence_tokens=sequence_tokens,
            auxiliary_embeddings=conditions,
            labels=None,
        )
        logits = _forward_output.structure_logits
        logits = self.logits_parameterization(logits=logits, xt=xt)
       
        if shield_special_tokens:
            for i in range(C.VQVAE_CODEBOOK_SIZE, C.VQVAE_CODEBOOK_SIZE + 5):
                logits[..., i] += self.neg_infinity

        if self.sequence_prediction:
            sequence_logits = _forward_output.sequence_logits
            return logits, sequence_logits    
    
        return logits, None

    def q_xt(self, x, move_chance, condition_seq=None, non_moving_mask=None):
        """Computes the noisy sample xt.
        
        Args:
            x: long with shape (B, L)
            move_chance: float with shape (B, )
            condition_seq: long with shape (B, L)
        """
        # fully random masking
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        # if not None, mask the tokens other than the non-moving tokens
        # non_moving_mask: (B, L), 1 for non-moving tokens
        if non_moving_mask is not None:
            move_indices = move_indices & (~non_moving_mask)

        xt = torch.where(move_indices, self.mask_index, x)
        if self.coupled_condition_mask and condition_seq is not None:
            condition_seq = torch.where(move_indices, self.condition_mask_index, condition_seq)     # mask the sequence tokens as well, coupled position
        return xt, condition_seq

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t
    
    def logits_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits   # (B, L, V+1)

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    @torch.no_grad()
    def ddpm_sample(self, sequence_tokens, num_steps=None, eps=1e-5, input_prior=None, sample_max_t=1.0):
        """Generate samples from the model."""
        self.net.eval()
        self.noise.eval()
        self.sigma_embedder.eval()

        # Lightning auto-casting is not working in this method for some reason        
        if num_steps is None:
            print("Using by default num_steps: 1000")
            num_steps = 1000
        if input_prior is None:
            x = self._sample_prior(*sequence_tokens.shape).to(self.device)
            assert sample_max_t == 1.0, f"sample_max_t has to be 1.0 when input_prior is None"
        else:
            # partially masked tokens
            # round trip diffusion
            print(f"Using input_prior: {input_prior.shape}")
            x = input_prior.to(self.device)
            assert x.shape == sequence_tokens.shape, f"Invalid input_prior shape: {x.shape} v.s. (seq) {sequence_tokens.shape}"

        timesteps = torch.linspace(
            sample_max_t, eps, num_steps + 1, device=self.device
        )
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in tqdm(range(num_steps), desc="DDPM Sampling ..."):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            x = self._ddpm_update(x, t, sequence_tokens=sequence_tokens, dt=dt)

        if self.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma_t = self.noise(t)[0]
            x, _ = self._model_wrapper(x, sequence_tokens, sigma_t)
            x = x.argmax(dim=-1)
        
        return x
    
    def _ddpm_update(self, x, t, sequence_tokens, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        # conditional sampling
        log_p_x0, _ = self._model_wrapper(x, sequence_tokens, sigma_t)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]    
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x
