"""FedDWM defense-method module (partial research release).

Original experiment settings, attack generation/training/evaluation, dataset
preparation, model architecture, simulator and command-line experiment entry
point are omitted. This file does not reproduce the manuscript experiments.

Config has no default values: supply every required method parameter explicitly.
The caller supplies a model, clean reference loaders, client model updates and
same-round reference margin measurements. It must orchestrate initialization,
verification, accepted-update aggregation, migration triggers, compensation and
atomic model/key acceptance or rollback.

Core method implementations remain visible in this partial release.
"""
from __future__ import annotations
import hashlib
import hmac
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Config:
    watermark_bits: int
    accept_threshold: float
    candidate_quantile: float
    migration_rate: float
    migration_warning_threshold: float
    watermark_param_fraction: float
    watermark_param_count: Optional[int]
    watermark_carrier_scope: str
    classifier_weight_name: str
    watermark_lambda: float
    watermark_lr: float
    embed_threshold: float
    task_loss_tolerance: float
    embed_check_interval: int
    candidate_quantile_step: float
    candidate_quantile_max: float
    projection_distribution: str
    normalize_projection_by_width: bool
    watermark_loss_reduction: str
    importance_mode: str
    importance_microbatch_size: int
    importance_max_samples: Optional[int]
    mixture_max_components: int
    mixture_min_component_size: int
    mixture_min_bic_improvement: float
    mixture_eps: float
    server_master_key: str

@dataclass(frozen=True)
class ParameterEntry:
    name: str
    start: int
    end: int
    shape: tuple[int, ...]

@dataclass(frozen=True)
class ParameterLayout:
    entries: tuple[ParameterEntry, ...]
    total_numel: int
    signature: str

    @classmethod
    def from_model(cls, model: nn.Module) -> 'ParameterLayout':
        entries: list[ParameterEntry] = []
        offset = 0
        signature_parts: list[str] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or not parameter.is_floating_point():
                continue
            end = offset + parameter.numel()
            shape = tuple((int(x) for x in parameter.shape))
            entries.append(ParameterEntry(name, offset, end, shape))
            signature_parts.append(f'{name}:{shape}:{offset}:{end}')
            offset = end
        digest = hashlib.sha256('|'.join(signature_parts).encode('utf-8')).hexdigest()
        return cls(tuple(entries), offset, digest)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple((entry.name for entry in self.entries))

    def assert_compatible(self, model: nn.Module) -> None:
        other = ParameterLayout.from_model(model)
        if self.signature != other.signature:
            raise RuntimeError('Model parameter layout differs from the checkpoint/watermark layout')

    def flatten_model(self, model: nn.Module) -> torch.Tensor:
        named = dict(model.named_parameters())
        parameters = [named[entry.name] for entry in self.entries]
        return parameters_to_vector(parameters)

    def assign_model_vector(self, model: nn.Module, vector: torch.Tensor) -> None:
        named = dict(model.named_parameters())
        parameters = [named[entry.name] for entry in self.entries]
        vector_to_parameters(vector, parameters)

    def parameter_masks(self, positions: torch.Tensor, model: nn.Module) -> list[torch.Tensor]:
        if positions.device.type != 'cpu':
            positions = positions.cpu()
        flat_mask = torch.zeros(self.total_numel, dtype=torch.bool)
        flat_mask[positions.long()] = True
        named = dict(model.named_parameters())
        masks: list[torch.Tensor] = []
        for entry in self.entries:
            parameter = named[entry.name]
            masks.append(flat_mask[entry.start:entry.end].reshape(entry.shape).to(device=parameter.device, dtype=parameter.dtype))
        return masks

def watermark_eligible_positions(layout: ParameterLayout, cfg: Config) -> torch.Tensor:
    if cfg.watermark_carrier_scope == 'all':
        return torch.arange(layout.total_numel, dtype=torch.long)
    for entry in layout.entries:
        if entry.name == cfg.classifier_weight_name:
            return torch.arange(entry.start, entry.end, dtype=torch.long)
    raise ValueError(f'Classifier parameter {cfg.classifier_weight_name!r} was not found in the trainable-parameter layout')

@dataclass
class WatermarkKey:
    positions: torch.Tensor
    projection: torch.Tensor
    bits: torch.Tensor
    version: int
    seed_label: str

    def clone(self, *, version: Optional[int]=None) -> 'WatermarkKey':
        return WatermarkKey(self.positions.clone(), self.projection.clone(), self.bits.clone(), self.version if version is None else version, self.seed_label)

    def to_checkpoint(self) -> dict[str, Any]:
        return {'positions': self.positions.cpu(), 'projection': self.projection.cpu(), 'bits': self.bits.cpu(), 'version': self.version, 'seed_label': self.seed_label}

def derive_seed(master_key: str, label: str) -> int:
    digest = hmac.new(master_key.encode('utf-8'), label.encode('utf-8'), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], 'big') % (2 ** 63 - 1)

def make_signature_bits(cfg: Config) -> torch.Tensor:
    zeros = cfg.watermark_bits // 2
    bits = torch.cat([torch.zeros(zeros, dtype=torch.float32), torch.ones(cfg.watermark_bits - zeros, dtype=torch.float32)])
    generator = torch.Generator().manual_seed(derive_seed(cfg.server_master_key, 'bits'))
    return bits[torch.randperm(cfg.watermark_bits, generator=generator)]

def make_projection(cfg: Config, width: int, round_index: int, attempt: int) -> tuple[torch.Tensor, str]:
    label = f'proj|round={round_index}|attempt={attempt}'
    generator = torch.Generator().manual_seed(derive_seed(cfg.server_master_key, label))
    if cfg.projection_distribution == 'rademacher':
        projection = torch.randint(0, 2, (cfg.watermark_bits, width), generator=generator, dtype=torch.int8).float()
        projection.mul_(2).sub_(1)
    else:
        projection = torch.randn(cfg.watermark_bits, width, generator=generator, dtype=torch.float32)
    if cfg.normalize_projection_by_width:
        projection.div_(math.sqrt(width))
    return (projection.contiguous(), label)

def make_projection_columns_for_positions(cfg: Config, positions: torch.Tensor, total_width: int, round_index: int, attempt: int) -> tuple[torch.Tensor, str]:
    cpu_positions = positions.detach().cpu().long().flatten()
    if cpu_positions.numel() == 0:
        return (torch.empty(cfg.watermark_bits, 0, dtype=torch.float32), f'proj-new|round={round_index}|attempt={attempt}|count=0')
    columns: list[torch.Tensor] = []
    for position in cpu_positions.tolist():
        label = f'proj-col|round={round_index}|attempt={attempt}|position={position}'
        generator = torch.Generator().manual_seed(derive_seed(cfg.server_master_key, label))
        if cfg.projection_distribution == 'rademacher':
            column = torch.randint(0, 2, (cfg.watermark_bits, 1), generator=generator, dtype=torch.int8).float()
            column.mul_(2).sub_(1)
        else:
            column = torch.randn(cfg.watermark_bits, 1, generator=generator, dtype=torch.float32)
        if cfg.normalize_projection_by_width:
            column.div_(math.sqrt(total_width))
        columns.append(column)
    position_digest = hashlib.sha256(','.join((str(x) for x in cpu_positions.tolist())).encode('utf-8')).hexdigest()[:12]
    projection = torch.cat(columns, dim=1).contiguous()
    return (projection, f'proj-new|round={round_index}|attempt={attempt}|count={cpu_positions.numel()}|positions_sha256={position_digest}')

def resolve_watermark_width(cfg: Config, total_numel: int, eligible_numel: Optional[int]=None) -> int:
    carrier_numel = total_numel if eligible_numel is None else eligible_numel
    if cfg.watermark_param_count is not None:
        width = int(cfg.watermark_param_count)
    else:
        width = int(math.ceil(cfg.watermark_param_fraction * carrier_numel))
    if not 1 <= width <= carrier_numel // 2:
        raise ValueError(f'Resolved d_w={width}; require 1 <= d_w <= eligible_parameters/2 with eligible_parameters={carrier_numel} to leave enough positions for migration')
    return width

def stable_lowest_indices(values: torch.Tensor, count: int) -> torch.Tensor:
    values_np = values.detach().cpu().double().numpy()
    indices = np.arange(values_np.size, dtype=np.int64)
    order = np.lexsort((indices, values_np))
    return torch.from_numpy(order[:count].copy()).long()

def candidate_pool(importance: torch.Tensor, quantile: float, minimum_size: int, eligible_positions: Optional[torch.Tensor]=None) -> torch.Tensor:
    eligible = torch.arange(importance.numel(), dtype=torch.long) if eligible_positions is None else eligible_positions.detach().cpu().long()
    if eligible.numel() == 0:
        raise ValueError('Watermark carrier scope contains no eligible positions')
    count = max(minimum_size, int(math.ceil(quantile * eligible.numel())))
    count = min(count, eligible.numel())
    local = stable_lowest_indices(importance[eligible], count)
    return eligible[local]

def migration_selection_pool(importance: torch.Tensor, quantile: float, minimum_size: int, eligible_positions: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
    pool = candidate_pool(importance, quantile, minimum_size, eligible_positions)
    return (pool, -importance)

def compute_parameter_importance(model: nn.Module, loader: DataLoader[Any], layout: ParameterLayout, cfg: Config, device: torch.device) -> torch.Tensor:
    layout.assert_compatible(model)
    was_training = model.training
    model.eval()
    named = dict(model.named_parameters())
    parameters = [named[entry.name] for entry in layout.entries]
    importance = torch.zeros(layout.total_numel, device=device)
    observed = 0
    vmap_microbatch = cfg.importance_microbatch_size
    vectorized_grad: Optional[Any] = None
    functional_params: dict[str, torch.Tensor] = {}
    functional_buffers: dict[str, torch.Tensor] = {}
    if cfg.importance_mode == 'per_sample_vmap':
        functional_params = {entry.name: named[entry.name].detach() for entry in layout.entries}
        functional_buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        def single_sample_loss(current_params: Mapping[str, torch.Tensor], current_buffers: Mapping[str, torch.Tensor], image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            logits = functional_call(model, (current_params, current_buffers), (image.unsqueeze(0),), strict=True)
            return F.cross_entropy(logits, label.unsqueeze(0))
        vectorized_grad = vmap(grad(single_sample_loss), in_dims=(None, None, 0, 0), randomness='error')
    target_samples = len(loader.dataset)
    if cfg.importance_max_samples is not None:
        target_samples = min(target_samples, cfg.importance_max_samples)

    def accumulate_one(loss: torch.Tensor, weight: int=1) -> None:
        nonlocal importance, observed
        gradients = torch.autograd.grad(loss, parameters, retain_graph=False)
        flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
        importance.add_(flat.square(), alpha=float(weight))
        observed += weight
    for images, labels in loader:
        if observed >= target_samples:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        remaining = min(images.size(0), target_samples - observed)
        images = images[:remaining]
        labels = labels[:remaining]
        if cfg.importance_mode == 'per_sample_vmap':
            assert vectorized_grad is not None
            start = 0
            while start < remaining:
                end = min(start + vmap_microbatch, remaining)
                try:
                    gradients = vectorized_grad(functional_params, functional_buffers, images[start:end], labels[start:end])
                except torch.cuda.OutOfMemoryError:
                    if device.type != 'cuda' or vmap_microbatch <= 1:
                        raise
                    vmap_microbatch = max(1, vmap_microbatch // 2)
                    torch.cuda.empty_cache()
                    print(f'[显存调整] CUDA显存不足；已将参数重要性微批量降为{vmap_microbatch}并重试')
                    continue
                for entry in layout.entries:
                    importance[entry.start:entry.end].add_(gradients[entry.name].detach().square().sum(dim=0).reshape(-1))
                observed += end - start
                start = end
                del gradients
        elif cfg.importance_mode == 'per_sample':
            for sample_index in range(images.size(0)):
                logits = model(images[sample_index:sample_index + 1])
                loss = F.cross_entropy(logits, labels[sample_index:sample_index + 1])
                accumulate_one(loss)
        else:
            loss = F.cross_entropy(model(images), labels)
            accumulate_one(loss, weight=remaining)
    if observed == 0:
        raise RuntimeError('D_server loader produced no samples')
    if was_training:
        model.train()
    result = importance.div_(observed).cpu()
    return result

def build_initial_key(importance: torch.Tensor, layout: ParameterLayout, cfg: Config) -> WatermarkKey:
    eligible = watermark_eligible_positions(layout, cfg)
    width = resolve_watermark_width(cfg, layout.total_numel, eligible.numel())
    candidates = candidate_pool(importance, cfg.candidate_quantile, width, eligible)
    label = 'mask|round=0|attempt=0'
    generator = torch.Generator().manual_seed(derive_seed(cfg.server_master_key, label))
    selected = candidates[torch.randperm(candidates.numel(), generator=generator)[:width]]
    positions = torch.sort(selected).values
    projection, projection_label = make_projection(cfg, width, 0, 0)
    return WatermarkKey(positions, projection, make_signature_bits(cfg), version=0, seed_label=f'{label};{projection_label}')

def watermark_logits_from_vector(vector: torch.Tensor, key: WatermarkKey) -> torch.Tensor:
    positions = key.positions.to(vector.device)
    projection = key.projection.to(vector.device, dtype=vector.dtype)
    return projection @ vector[positions]

def watermark_loss(model: nn.Module, layout: ParameterLayout, key: WatermarkKey, cfg: Config) -> torch.Tensor:
    vector = layout.flatten_model(model)
    logits = watermark_logits_from_vector(vector, key)
    target = key.bits.to(logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, reduction=cfg.watermark_loss_reduction)

def watermark_score_from_logits(logits: torch.Tensor, bits: torch.Tensor) -> float:
    predicted = logits.detach().ge(0).cpu()
    target = bits.detach().ge(0.5).cpu()
    matches = int(predicted.eq(target).sum().item())
    return matches / int(target.numel())

@dataclass(frozen=True)
class WatermarkEvidence:
    bit_accuracy: float
    match_count: int
    logit_drift: float
    margin_drop: float
    masked_update_ratio: float
    mean_signed_margin: float

@dataclass
class DeviceWatermarkVerifier:
    positions: torch.Tensor
    projection: torch.Tensor
    target_bits: torch.Tensor
    cpu_projection: torch.Tensor
    cpu_bits: torch.Tensor
    reference_vector: Optional[torch.Tensor] = None
    reference_selected: Optional[torch.Tensor] = None
    reference_logits: Optional[torch.Tensor] = None
    reference_margins: Optional[torch.Tensor] = None
    reliable_bits: Optional[torch.Tensor] = None
    reference_logits_norm: Optional[torch.Tensor] = None
    reliable_margin_norm: Optional[torch.Tensor] = None
    has_reliable_bits: bool = False
    boundary_tolerance: float = 1e-06

    @classmethod
    def from_key(cls, key: WatermarkKey, device: torch.device, dtype: torch.dtype, reference_vector: Optional[torch.Tensor]=None) -> 'DeviceWatermarkVerifier':
        positions = key.positions.to(device=device, dtype=torch.long)
        projection = key.projection.to(device=device, dtype=dtype)
        target_bits = key.bits.to(device=device).ge(0.5)
        reference_selected: Optional[torch.Tensor] = None
        reference_logits: Optional[torch.Tensor] = None
        reference_margins: Optional[torch.Tensor] = None
        reliable_bits: Optional[torch.Tensor] = None
        reference_logits_norm: Optional[torch.Tensor] = None
        reliable_margin_norm: Optional[torch.Tensor] = None
        has_reliable_bits = False
        if reference_vector is not None:
            reference_selected = reference_vector[positions]
            reference_logits = projection @ reference_selected
            signs = target_bits.to(dtype=dtype).mul(2).sub(1)
            reference_margins = signs * reference_logits
            reliable_bits = reference_margins.gt(0)
            reference_logits_norm = reference_logits.norm()
            has_reliable_bits = bool(reliable_bits.any().item())
            if has_reliable_bits:
                reliable_margin_norm = reference_margins[reliable_bits].norm()
        return cls(positions=positions, projection=projection, target_bits=target_bits, cpu_projection=key.projection.detach().cpu(), cpu_bits=key.bits.detach().cpu(), reference_vector=reference_vector, reference_selected=reference_selected, reference_logits=reference_logits, reference_margins=reference_margins, reliable_bits=reliable_bits, reference_logits_norm=reference_logits_norm, reliable_margin_norm=reliable_margin_norm, has_reliable_bits=has_reliable_bits)

    @torch.no_grad()
    def _score_logits(self, logits: torch.Tensor, selected: torch.Tensor) -> float:
        matches_tensor = logits.ge(0).eq(self.target_bits).sum()
        if logits.device.type == 'cuda' and logits.numel() > 0:
            stats = torch.stack([matches_tensor.to(logits.dtype), logits.abs().min()]).cpu()
            matches = int(stats[0].item())
            near_boundary = float(stats[1].item()) < self.boundary_tolerance
        else:
            matches = int(matches_tensor.item())
            near_boundary = False
        if near_boundary:
            cpu_logits = self.cpu_projection.to(dtype=selected.dtype) @ selected.cpu()
            return watermark_score_from_logits(cpu_logits, self.cpu_bits)
        return matches / int(self.target_bits.numel())

    @torch.no_grad()
    def score_vector(self, vector: torch.Tensor) -> float:
        selected = vector[self.positions]
        logits = self.projection @ selected
        return self._score_logits(logits, selected)

    @torch.no_grad()
    def evidence_vector(self, vector: torch.Tensor) -> WatermarkEvidence:
        selected = vector[self.positions]
        logits = self.projection @ selected
        bit_accuracy = self._score_logits(logits, selected)
        match_count = int(round(bit_accuracy * self.target_bits.numel()))
        signs = self.target_bits.to(dtype=logits.dtype).mul(2).sub(1)
        margins = signs * logits
        if self.reference_vector is None or self.reference_selected is None or self.reference_logits is None or (self.reference_margins is None) or (self.reliable_bits is None) or (self.reference_logits_norm is None):
            return WatermarkEvidence(bit_accuracy, match_count, float('nan'), float('nan'), float('nan'), float(margins.mean().item()))
        eps = torch.finfo(logits.dtype).eps
        logit_drift = (logits - self.reference_logits).norm() / (self.reference_logits_norm + eps)
        if self.has_reliable_bits:
            assert self.reliable_margin_norm is not None
            margin_loss = F.relu(self.reference_margins[self.reliable_bits] - margins[self.reliable_bits])
            margin_drop = margin_loss.norm() / (self.reliable_margin_norm + eps)
        else:
            margin_drop = torch.zeros((), device=vector.device, dtype=vector.dtype)
        update = vector - self.reference_vector
        update_norm = update.norm()
        masked_norm = (selected - self.reference_selected).norm()
        masked_update_ratio = torch.where(update_norm > eps, masked_norm / update_norm, torch.zeros_like(update_norm))
        values = torch.stack([logit_drift, margin_drop, masked_update_ratio, margins.mean()]).cpu()
        return WatermarkEvidence(bit_accuracy=bit_accuracy, match_count=match_count, logit_drift=float(values[0].item()), margin_drop=float(values[1].item()), masked_update_ratio=float(values[2].item()), mean_signed_margin=float(values[3].item()))

@torch.no_grad()
def watermark_score_model(model: nn.Module, layout: ParameterLayout, key: WatermarkKey) -> float:
    vector = layout.flatten_model(model)
    logits = watermark_logits_from_vector(vector, key)
    return watermark_score_from_logits(logits, key.bits)

@torch.no_grad()
def evaluate_task_loss(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        total_loss += float(F.cross_entropy(model(images), labels, reduction='sum').item())
        total += labels.numel()
    if was_training:
        model.train()
    if total == 0:
        raise RuntimeError('Evaluation loader produced no samples')
    return total_loss / total

@dataclass
class EmbedResult:
    accepted: bool
    steps: int
    watermark_score: float
    task_loss: float
    task_loss_delta: float

def masked_watermark_compensation(model: nn.Module, baseline_task_loss: float, key: WatermarkKey, layout: ParameterLayout, loader: DataLoader[Any], cfg: Config, device: torch.device, max_steps: int) -> EmbedResult:
    layout.assert_compatible(model)
    model.eval()
    named = dict(model.named_parameters())
    parameters = [named[entry.name] for entry in layout.entries]
    masks = layout.parameter_masks(key.positions, model)
    iterator: Iterator[Any] = iter(loader)
    last_score = watermark_score_model(model, layout, key)
    last_task_loss = evaluate_task_loss(model, loader, device)
    if last_score >= cfg.embed_threshold and last_task_loss - baseline_task_loss <= cfg.task_loss_tolerance:
        return EmbedResult(True, 0, last_score, last_task_loss, last_task_loss - baseline_task_loss)
    for step in range(1, max_steps + 1):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        task = F.cross_entropy(model(images), labels)
        total = task + cfg.watermark_lambda * watermark_loss(model, layout, key, cfg)
        total.backward()
        with torch.no_grad():
            for parameter, mask in zip(parameters, masks):
                if parameter.grad is not None:
                    parameter.add_(parameter.grad * mask, alpha=-cfg.watermark_lr)
        if step % cfg.embed_check_interval == 0 or step == max_steps:
            last_score = watermark_score_model(model, layout, key)
            last_task_loss = evaluate_task_loss(model, loader, device)
            delta = last_task_loss - baseline_task_loss
            if last_score >= cfg.embed_threshold and delta <= cfg.task_loss_tolerance:
                return EmbedResult(True, step, last_score, last_task_loss, delta)
    return EmbedResult(False, max_steps, last_score, last_task_loss, last_task_loss - baseline_task_loss)

@dataclass
class MigrationProposal:
    key: WatermarkKey
    candidate_quantile: float
    keep_count: int
    move_count: int
    stay_ratio: float

def build_migration_key(old_key: WatermarkKey, importance: torch.Tensor, layout: ParameterLayout, cfg: Config, round_index: int, attempt: int) -> Optional[MigrationProposal]:
    width = old_key.positions.numel()
    quantile = cfg.candidate_quantile
    old_positions = old_key.positions.cpu()
    eligible = watermark_eligible_positions(layout, cfg)
    minimum_candidate_size = min(eligible.numel(), 2 * width)
    candidate, preference = migration_selection_pool(importance, quantile, minimum_candidate_size, eligible)
    while True:
        candidate_membership = torch.zeros(importance.numel(), dtype=torch.bool)
        candidate_membership[candidate] = True
        stable = old_positions[candidate_membership[old_positions]]
        minimum_move = int(math.ceil(cfg.migration_rate * width))
        keep_count = min(width - minimum_move, stable.numel())
        if keep_count:
            stable_preference = preference[stable]
            stable_order = np.lexsort((stable.numpy(), -stable_preference.double().numpy()))
            keep = stable[torch.from_numpy(stable_order[:keep_count].copy()).long()]
        else:
            keep = torch.empty(0, dtype=torch.long)
        move_count = width - keep_count
        old_membership = torch.zeros(importance.numel(), dtype=torch.bool)
        old_membership[old_positions] = True
        available = candidate[~old_membership[candidate]]
        if available.numel() >= move_count:
            break
        if quantile >= cfg.candidate_quantile_max - 1e-12:
            return None
        quantile = min(cfg.candidate_quantile_max, quantile + cfg.candidate_quantile_step)
        candidate, preference = migration_selection_pool(importance, quantile, minimum_candidate_size, eligible)
    mask_label = f'mask|round={round_index}|attempt={attempt}'
    generator = torch.Generator().manual_seed(derive_seed(cfg.server_master_key, mask_label))
    new_positions = available[torch.randperm(available.numel(), generator=generator)[:move_count]]
    if keep_count:
        old_columns = torch.searchsorted(old_positions, keep)
        if not torch.equal(old_positions[old_columns], keep):
            raise RuntimeError('Kept watermark positions lost their old column mapping')
        keep_projection = old_key.projection[:, old_columns].clone()
    else:
        keep_projection = torch.empty(cfg.watermark_bits, 0, dtype=old_key.projection.dtype)
    new_projection, projection_label = make_projection_columns_for_positions(cfg, new_positions, width, round_index, attempt)
    combined_positions = torch.cat([keep, new_positions])
    combined_projection = torch.cat([keep_projection, new_projection], dim=1)
    order = torch.argsort(combined_positions)
    positions = combined_positions[order]
    projection = combined_projection[:, order].contiguous()
    if positions.numel() != width or projection.shape != (cfg.watermark_bits, width):
        raise RuntimeError('Migrated watermark key has inconsistent dimensions')
    new_key = WatermarkKey(positions, projection, old_key.bits.clone(), version=round_index, seed_label=f'{mask_label};preserved_columns={keep_count};{projection_label}')
    stay_ratio = float(stable.numel() / width)
    return MigrationProposal(new_key, quantile, keep_count, move_count, stay_ratio)

def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

def clone_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}

@dataclass
class ClientUpdate:
    client_id: int
    state: Optional[dict[str, torch.Tensor]]
    num_samples: int
    watermark_score: float = 0.0
    watermark_match_count: int = 0
    wm_logit_drift: float = float('nan')
    wm_margin_drop: float = float('nan')
    masked_update_ratio: float = float('nan')
    mean_signed_margin: float = float('nan')
    stage1_accepted: bool = False
    reference_mixture_anomaly: bool = False
    reject_reason: str = 'none'
    accepted: bool = False

@dataclass(frozen=True)
class ReferenceMixtureStats:
    stage1_count: int
    valid_client_count: int
    reference_count: int
    reference_median: float
    component_count: int
    component_centers: tuple[float, ...]
    component_sizes: tuple[int, ...]
    bic_values: tuple[float, ...]
    anchor_component: int
    accepted_count: int
    available: bool
    reason: str

def _optimal_1d_partition(values: Sequence[float], components: int, minimum_size: int, eps: float) -> Optional[tuple[list[int], tuple[float, ...], tuple[int, ...], float]]:
    n = len(values)
    if components < 1 or n < components * minimum_size:
        return None
    order = sorted(range(n), key=lambda index: (values[index], index))
    sorted_values = [float(values[index]) for index in order]
    prefix = [0.0]
    prefix_sq = [0.0]
    for value in sorted_values:
        prefix.append(prefix[-1] + value)
        prefix_sq.append(prefix_sq[-1] + value * value)

    def interval_sse(start: int, end: int) -> float:
        count = end - start
        total = prefix[end] - prefix[start]
        total_sq = prefix_sq[end] - prefix_sq[start]
        return max(0.0, total_sq - total * total / count)
    infinity = float('inf')
    dp = [[infinity] * (n + 1) for _ in range(components + 1)]
    previous = [[-1] * (n + 1) for _ in range(components + 1)]
    dp[0][0] = 0.0
    for k in range(1, components + 1):
        minimum_end = k * minimum_size
        for end in range(minimum_end, n + 1):
            start_min = (k - 1) * minimum_size
            start_max = end - minimum_size
            for start in range(start_min, start_max + 1):
                candidate = dp[k - 1][start] + interval_sse(start, end)
                if candidate < dp[k][end] - eps:
                    dp[k][end] = candidate
                    previous[k][end] = start
    if not math.isfinite(dp[components][n]):
        return None
    bounds: list[tuple[int, int]] = []
    end = n
    for k in range(components, 0, -1):
        start = previous[k][end]
        if start < 0:
            return None
        bounds.append((start, end))
        end = start
    bounds.reverse()
    labels = [-1] * n
    centers: list[float] = []
    sizes: list[int] = []
    for label, (start, end) in enumerate(bounds):
        members = sorted_values[start:end]
        centers.append(sum(members) / len(members))
        sizes.append(len(members))
        for sorted_index in range(start, end):
            labels[order[sorted_index]] = label
    return (labels, tuple(centers), tuple(sizes), max(dp[components][n], eps))

def compute_reference_mixture_stats(updates: Sequence[ClientUpdate], reference_margins: Sequence[float], cfg: Config) -> tuple[ReferenceMixtureStats, dict[int, int]]:
    stage1 = [update for update in updates if update.stage1_accepted]
    valid = [update for update in stage1 if math.isfinite(update.wm_margin_drop) and update.wm_margin_drop >= 0]
    references = sorted((float(value) for value in reference_margins if math.isfinite(value) and value >= 0))
    empty = ReferenceMixtureStats(stage1_count=len(stage1), valid_client_count=len(valid), reference_count=len(references), reference_median=float('nan'), component_count=0, component_centers=(), component_sizes=(), bic_values=(), anchor_component=-1, accepted_count=0, available=False, reason='not_evaluated')
    if len(references) < 3:
        return (replace(empty, reason='insufficient_server_references'), {})
    if not valid:
        return (replace(empty, reference_count=len(references), reason='no_valid_clients'), {})
    reference_median = float(np.median(np.asarray(references, dtype=np.float64)))
    values = [float(update.wm_margin_drop) for update in valid]
    partitions: list[tuple[list[int], tuple[float, ...], tuple[int, ...], float]] = []
    bic_values: list[float] = []
    maximum_components = min(cfg.mixture_max_components, len(values))
    for components in range(1, maximum_components + 1):
        minimum_size = 1 if components == 1 else cfg.mixture_min_component_size
        partition = _optimal_1d_partition(values, components, minimum_size, cfg.mixture_eps)
        if partition is None:
            break
        partitions.append(partition)
        sse = partition[3]
        parameter_count = 2 * components
        bic_values.append(len(values) * math.log(sse / len(values) + cfg.mixture_eps) + parameter_count * math.log(max(2, len(values))))
    if not partitions:
        return (replace(empty, reason='mixture_fit_failed'), {})
    best_index = min(range(len(bic_values)), key=bic_values.__getitem__)
    if best_index > 0 and bic_values[0] - bic_values[best_index] < cfg.mixture_min_bic_improvement:
        best_index = 0
    labels, centers, sizes, _ = partitions[best_index]
    anchor_component = min(range(len(centers)), key=lambda index: (abs(centers[index] - reference_median), centers[index]))
    assignments = {update.client_id: labels[index] for index, update in enumerate(valid)}
    available = True
    reason = 'reference_anchored'
    accepted_count = sum((labels[index] == anchor_component for index in range(len(valid))))
    stats = ReferenceMixtureStats(stage1_count=len(stage1), valid_client_count=len(valid), reference_count=len(references), reference_median=reference_median, component_count=best_index + 1, component_centers=centers, component_sizes=sizes, bic_values=tuple(bic_values), anchor_component=anchor_component, accepted_count=accepted_count, available=available, reason=reason)
    return (stats, assignments)

def apply_reference_mixture_decisions(updates: Sequence[ClientUpdate], stats: ReferenceMixtureStats, assignments: Mapping[int, int]) -> None:
    for update in updates:
        update.reference_mixture_anomaly = False
        if not update.stage1_accepted:
            update.accepted = False
            update.reject_reason = 'bit_accuracy'
        elif not math.isfinite(update.wm_margin_drop) or update.wm_margin_drop < 0:
            update.accepted = False
            update.reference_mixture_anomaly = True
            update.reject_reason = 'invalid_margin'
        elif not stats.available:
            update.accepted = False
            update.reference_mixture_anomaly = True
            update.reject_reason = 'reference_mixture_unavailable'
        elif assignments.get(update.client_id, -1) != stats.anchor_component:
            update.accepted = False
            update.reference_mixture_anomaly = True
            update.reject_reason = 'non_reference_component'
        else:
            update.accepted = True
            update.reject_reason = 'none'
        if not update.accepted:
            update.state = None

def near_threshold_client_statistics(updates: Sequence[ClientUpdate], cfg: Config) -> tuple[int, float]:
    count = sum((math.isfinite(update.watermark_score) and cfg.migration_warning_threshold <= update.watermark_score < cfg.accept_threshold for update in updates))
    ratio = count / len(updates) if updates else 0.0
    return (count, ratio)

def aggregate_states(updates: Sequence[ClientUpdate]) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError('Cannot aggregate an empty update list')
    total_samples = sum((update.num_samples for update in updates))
    if total_samples <= 0:
        raise ValueError('Accepted updates have no samples')
    if any((update.state is None for update in updates)):
        raise ValueError('An accepted update is missing its retained model state')
    states = [update.state for update in updates]
    assert all((state is not None for state in states))
    result: dict[str, torch.Tensor] = {}
    first_state = states[0]
    assert first_state is not None
    keys = list(first_state.keys())
    for name in keys:
        tensors = [state[name] for state in states if state is not None]
        first = tensors[0]
        if first.is_floating_point() or first.is_complex():
            accumulator = torch.zeros_like(first, dtype=torch.float64)
            for update, tensor in zip(updates, tensors):
                accumulator.add_(tensor.double(), alpha=update.num_samples / total_samples)
            result[name] = accumulator.to(dtype=first.dtype)
        else:
            stacked = torch.stack([tensor.long() for tensor in tensors])
            result[name] = stacked.max(dim=0).values.to(dtype=first.dtype)
    return result

@dataclass
class ServerState:
    version: int
    model_state: dict[str, torch.Tensor]
    key: WatermarkKey
    last_migration_round: Optional[int] = None

    def clone(self) -> 'ServerState':
        return ServerState(self.version, clone_state_dict(self.model_state), self.key.clone(), self.last_migration_round)
