"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import inspect
import time
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lavis.models import load_model_and_preprocess

from data_utils import (
    OACIRRDataset,
    FashionIQDataset,
    CIRRDataset,
    squarepad_transform,
    targetpad_transform,
)
from utils import (
    extract_index_blip_features,
    collate_fn,
    custom_collate_fn,
    device,
)


@torch.no_grad()
def extract_index_blip_features_with_raw(
    classic_val_dataset,
    blip_model,
    save_memory: bool = False,
    batch_size: int = 32,
    num_workers: int = 6,
):
    """
    Extract gallery features for latent matching.

    Returns:
        index_features:
            [N, num_query_token, embed_dim], target retrieval features.
        index_raw_embeds:
            [N, num_visual_tokens, vision_width], raw visual_encoder outputs before ln_vision.
        index_names:
            list[str]
    """
    print(f"Extracting {classic_val_dataset.__class__.__name__} [{getattr(classic_val_dataset, 'variant', '')}] index features with raw visual tokens")

    classic_val_loader = DataLoader(
        dataset=classic_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    if not hasattr(blip_model, "extract_target_features_with_raw"):
        raise AttributeError(
            "The current model does not implement extract_target_features_with_raw(). "
            "Please use blip2_qformer_oacir_latent.py with this method."
        )

    index_features = []
    index_raw_embeds = []
    index_names = []

    blip_model.eval()

    for names, images in tqdm(classic_val_loader, ncols=140, ascii=True):
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            image_features, raw_embeds = blip_model.extract_target_features_with_raw(images)

        if save_memory:
            index_features.append(image_features.detach().cpu())
        else:
            index_features.append(image_features.detach())

        # raw visual tokens are large; keep them on CPU fp16 by default.
        index_raw_embeds.append(raw_embeds.detach().cpu().to(torch.float16))
        index_names.extend(names)

    index_features = torch.cat(index_features, dim=0)
    index_raw_embeds = torch.cat(index_raw_embeds, dim=0)

    return index_features, index_raw_embeds, index_names


@torch.no_grad()
def extract_index_blip_features_from_raw_cache(
    classic_val_dataset,
    blip_model,
    cache_path,
    save_memory: bool = False,
    batch_size: int = 32,
):
    """
    Load frozen-ViT gallery tokens from disk and rebuild features with the
    current epoch's ln_vision/Q-Former/projection parameters.

    The cache may contain extra images, but all images required by the current
    validation subset must be present. Returned tensors always follow the
    classic dataset's gallery order.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Validation feature cache not found: {cache_path}. "
            "Generate it with tools/precompute_oacirr_train_visual_embeds.py "
            "using --split val."
        )

    if not hasattr(blip_model, "extract_target_features_from_raw"):
        raise AttributeError(
            "The current model does not implement "
            "extract_target_features_from_raw()."
        )

    print(f"Loading validation raw visual embeddings from: {cache_path}")
    load_kwargs = {"map_location": "cpu"}
    if "mmap" in inspect.signature(torch.load).parameters:
        load_kwargs["mmap"] = True
    cache = torch.load(cache_path, **load_kwargs)
    cache_names = list(cache["names"])
    cache_embeds = cache["embeds"]
    cache_meta = cache.get("meta", {})

    if cache_meta.get("split") not in (None, "val"):
        raise ValueError(
            f"Expected a val cache, but cache split is {cache_meta.get('split')!r}"
        )

    expected_variant = getattr(classic_val_dataset, "variant", None)
    cached_variant = cache_meta.get("dataset")
    if cached_variant is not None and expected_variant is not None:
        if cached_variant != expected_variant:
            raise ValueError(
                f"Validation cache variant mismatch: cache={cached_variant}, "
                f"dataset={expected_variant}"
            )

    if len(cache_names) != len(set(cache_names)):
        raise ValueError(f"Validation cache contains duplicate image names: {cache_path}")

    dataset_names = list(classic_val_dataset.name_to_relpath.keys())
    cache_name_set = set(cache_names)
    missing_names = [name for name in dataset_names if name not in cache_name_set]
    if missing_names:
        preview = ", ".join(missing_names[:5])
        raise KeyError(
            f"Validation cache is missing {len(missing_names)} gallery images "
            f"for {expected_variant}: {preview}"
        )

    if len(cache_names) == len(dataset_names) and cache_name_set == set(dataset_names):
        # Retrieval metrics do not depend on gallery order. Keeping the cache's
        # native order avoids duplicating the multi-GB raw embedding tensor.
        desired_names = cache_names
        index_raw_embeds = cache_embeds
    else:
        desired_names = dataset_names
        name_to_cache_idx = {name: idx for idx, name in enumerate(cache_names)}
        cache_indices = torch.tensor(
            [name_to_cache_idx[name] for name in desired_names],
            dtype=torch.long,
        )
        index_raw_embeds = cache_embeds.index_select(0, cache_indices)

    if index_raw_embeds.dtype != torch.float16:
        index_raw_embeds = index_raw_embeds.to(dtype=torch.float16)
    del cache, cache_embeds

    index_features = []
    blip_model.eval()
    for start in tqdm(
        range(0, len(desired_names), batch_size),
        desc=f"Building [{expected_variant}] gallery features from raw cache",
        ncols=140,
        ascii=True,
    ):
        raw_batch = index_raw_embeds[start:start + batch_size]
        image_features = blip_model.extract_target_features_from_raw(raw_batch)
        if save_memory:
            image_features = image_features.detach().cpu()
        else:
            image_features = image_features.detach()
        index_features.append(image_features)

    index_features = torch.cat(index_features, dim=0)
    print(
        f"Loaded {len(desired_names)} cached gallery embeddings, "
        f"raw dtype={index_raw_embeds.dtype}"
    )
    return index_features, index_raw_embeds, desired_names


def compute_fiq_val_metrics(
    relative_val_dataset: FashionIQDataset,
    blip_model,
    index_features: torch.Tensor,
    index_names: List[str],
    txt_processors,
    save_memory: bool = False,
    eval_batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ.
    """
    pred_sim, target_names, reference_names, captions_all = generate_fiq_val_predictions(
        blip_model,
        relative_val_dataset,
        index_names,
        index_features,
        txt_processors,
        save_memory,
        eval_batch_size,
    )

    print(f"Computing FashionIQ {relative_val_dataset.dress_types} validation metrics...")

    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1)
    )

    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(len(target_names)).int(),
    )

    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(
    blip_model,
    relative_val_dataset: FashionIQDataset,
    index_names: List[str],
    index_features,
    txt_processors,
    save_memory: bool = False,
    eval_batch_size: int = 32,
):
    """
    Generate FashionIQ validation predictions.
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=eval_batch_size,
        num_workers=6,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )

    name_to_feat = dict(zip(index_names, index_features[-1]))

    target_names = []
    reference_names_all = []
    distance = []
    captions_all = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader, ncols=140, ascii=True):
        if len(captions) == 2 and isinstance(captions[0], (tuple, list)):
            captions = list(zip(*captions))

        input_captions = [
            f"{texts[0].strip('.?, ').capitalize()} and {texts[1].strip('.?, ')}"
            for texts in captions
        ]
        input_captions = [txt_processors["eval"](caption) for caption in input_captions]

        with torch.no_grad():
            if len(input_captions) == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(name_to_feat))

            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(device)

            reference_image_features = reference_image_features.to(device)

            batch_distance = blip_model.inference(
                reference_image_features,
                feature_curr,
                input_captions,
            )

            distance.append(batch_distance.cpu())
            captions_all += input_captions
            target_names.extend(batch_target_names)
            reference_names_all.extend(reference_names)

    distance = torch.vstack(distance)

    return distance, target_names, reference_names_all, captions_all


def compute_cirr_val_metrics(
    relative_val_dataset: CIRRDataset,
    blip_model,
    index_features,
    index_names: List[str],
    txt_processors,
    eval_batch_size: int = 32,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR.
    """
    pred_sim, reference_names, target_names, group_members, captions_all = generate_cirr_val_predictions(
        blip_model,
        relative_val_dataset,
        index_names,
        index_features,
        txt_processors,
        eval_batch_size,
    )

    print("Computing CIRR validation metrics...")

    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    reference_mask = torch.tensor(
        sorted_index_names
        != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0],
        sorted_index_names.shape[1] - 1,
    )

    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1)
    )

    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(len(target_names)).int(),
    )
    assert torch.equal(
        torch.sum(group_labels, dim=-1).int(),
        torch.ones(len(target_names)).int(),
    )

    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return (
        group_recall_at1,
        group_recall_at2,
        group_recall_at3,
        recall_at1,
        recall_at5,
        recall_at10,
        recall_at50,
    )


def generate_cirr_val_predictions(
    blip_model,
    relative_val_dataset: CIRRDataset,
    index_names: List[str],
    index_features,
    txt_processors,
    eval_batch_size: int = 32,
):
    """
    Generate CIRR validation predictions.
    """
    print("Compute CIRR validation predictions")

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=eval_batch_size,
        num_workers=6,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )

    name_to_feat = dict(zip(index_names, index_features[1]))

    distance = []
    target_names = []
    group_members = []
    reference_names = []
    captions_all = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
        relative_val_loader,
        ncols=140,
        ascii=True,
    ):
        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]

        with torch.no_grad():
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))

            reference_image_features = reference_image_features.to(device)

            batch_distance = blip_model.inference(
                reference_image_features,
                index_features[0],
                captions,
            )

            distance.append(batch_distance.cpu())
            captions_all += captions
            target_names.extend(batch_target_names)
            group_members.extend(batch_group_members)
            reference_names.extend(batch_reference_names)

    distance = torch.vstack(distance)

    return distance, reference_names, target_names, group_members, captions_all


def compute_class_recall(
    sorted_index_names: np.ndarray,
    target_names: List[str],
    name_to_relpath: dict,
) -> Tuple[float, float, float]:
    """
    Compute OACIRR instance-level recall, i.e. R_ID@K.
    """
    name_to_folder = {
        name: path.split("/")[-2]
        for name, path in name_to_relpath.items()
    }

    target_folders = np.array([name_to_folder.get(name) for name in target_names])
    folder_mapper = np.vectorize(name_to_folder.get)
    sorted_index_folders = folder_mapper(sorted_index_names)

    class_labels = (sorted_index_folders == target_folders[:, np.newaxis]).astype(int)

    modified_class_labels = (
        (class_labels == 1)
        & (np.cumsum(class_labels, axis=1) == 1)
    ).astype(int)

    class_recall_at1 = (np.sum(modified_class_labels[:, :1]) / len(target_names)) * 100
    class_recall_at3 = (np.sum(modified_class_labels[:, :3]) / len(target_names)) * 100
    class_recall_at5 = (np.sum(modified_class_labels[:, :5]) / len(target_names)) * 100

    return class_recall_at1, class_recall_at3, class_recall_at5


def compute_oacirr_val_metrics(
    relative_val_dataset: OACIRRDataset,
    blip_model,
    index_features,
    index_names: List[str],
    txt_processors,
    highlight_inference: bool = False,
    save_results: bool = False,
    save_memory: bool = False,
    eval_batch_size: int = 32,
):
    """
    Compute OACIRR validation metrics with the original inference path.
    """
    (
        pred_sim,
        reference_names,
        target_names,
        modification_texts,
        reference_bboxes,
        activation_scalars,
    ) = generate_oacirr_val_predictions(
        blip_model,
        relative_val_dataset,
        index_names,
        index_features,
        txt_processors,
        highlight_inference,
        save_memory,
        eval_batch_size,
    )

    return _compute_oacirr_metrics_from_predictions(
        relative_val_dataset=relative_val_dataset,
        pred_sim=pred_sim,
        index_names=index_names,
        reference_names=reference_names,
        target_names=target_names,
        modification_texts=modification_texts,
        reference_bboxes=reference_bboxes,
        activation_scalars=activation_scalars,
        save_results=save_results,
    )


def compute_oacirr_val_metrics_latent(
    relative_val_dataset: OACIRRDataset,
    blip_model,
    index_features,
    index_raw_embeds,
    index_names: List[str],
    txt_processors,
    save_results: bool = False,
    save_memory: bool = False,
    latent_gallery_chunk_size: int = 1024,
    eval_batch_size: int = 32,
):
    """
    Compute OACIRR validation metrics with target-side latent instance matching.
    """
    (
        pred_sim,
        reference_names,
        target_names,
        modification_texts,
        reference_bboxes,
        activation_scalars,
    ) = generate_oacirr_val_predictions_latent(
        blip_model=blip_model,
        relative_val_dataset=relative_val_dataset,
        index_names=index_names,
        index_features=index_features,
        index_raw_embeds=index_raw_embeds,
        txt_processors=txt_processors,
        save_memory=save_memory,
        latent_gallery_chunk_size=latent_gallery_chunk_size,
        eval_batch_size=eval_batch_size,
    )

    return _compute_oacirr_metrics_from_predictions(
        relative_val_dataset=relative_val_dataset,
        pred_sim=pred_sim,
        index_names=index_names,
        reference_names=reference_names,
        target_names=target_names,
        modification_texts=modification_texts,
        reference_bboxes=reference_bboxes,
        activation_scalars=activation_scalars,
        save_results=save_results,
    )


def _compute_oacirr_metrics_from_predictions(
    relative_val_dataset: OACIRRDataset,
    pred_sim,
    index_names: List[str],
    reference_names: List[str],
    target_names: List[str],
    modification_texts: List[str],
    reference_bboxes: List,
    activation_scalars,
    save_results: bool = False,
):
    """
    Shared OACIRR metric computation.
    """
    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation metrics...")

    if pred_sim.size(0) != len(target_names) or pred_sim.size(1) != len(index_names):
        raise ValueError(
            "Prediction matrix shape does not match query/gallery metadata: "
            f"pred_sim={tuple(pred_sim.shape)}, queries={len(target_names)}, "
            f"gallery={len(index_names)}"
        )

    name_to_index = {name: idx for idx, name in enumerate(index_names)}
    try:
        reference_indices = torch.tensor(
            [name_to_index[name] for name in reference_names],
            dtype=torch.long,
            device=pred_sim.device,
        )
        target_indices = torch.tensor(
            [name_to_index[name] for name in target_names],
            dtype=torch.long,
            device=pred_sim.device,
        )
    except KeyError as error:
        raise KeyError(f"Validation image is missing from gallery: {error.args[0]}") from error

    # The metrics only use ranks up to 50. Excluding the reference in-place and
    # selecting Top-50 avoids full argsort tensors and a huge string matrix.
    query_indices = torch.arange(pred_sim.size(0), device=pred_sim.device)
    pred_sim[query_indices, reference_indices] = -torch.inf
    if pred_sim.size(1) < 2:
        raise ValueError("OACIRR metrics require at least two gallery images")
    metric_topk = min(50, pred_sim.size(1) - 1)
    top_index_chunks = []
    metric_query_chunk_size = 256
    for start in range(0, pred_sim.size(0), metric_query_chunk_size):
        end = min(start + metric_query_chunk_size, pred_sim.size(0))
        top_index_chunks.append(
            torch.topk(
                pred_sim[start:end],
                k=metric_topk,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices.cpu()
        )
    top_indices = torch.cat(top_index_chunks, dim=0)
    target_indices = target_indices.cpu().unsqueeze(1)
    labels = top_indices.eq(target_indices)

    name_to_folder = {
        name: path.split("/")[-2]
        for name, path in relative_val_dataset.name_to_relpath.items()
    }
    folder_to_index = {
        folder: idx
        for idx, folder in enumerate(dict.fromkeys(name_to_folder.values()))
    }
    gallery_folder_indices = torch.tensor(
        [folder_to_index[name_to_folder[name]] for name in index_names],
        dtype=torch.long,
    )
    target_folder_indices = torch.tensor(
        [folder_to_index[name_to_folder[name]] for name in target_names],
        dtype=torch.long,
    ).unsqueeze(1)
    class_labels = gallery_folder_indices[top_indices].eq(target_folder_indices)

    def recall_at(labels_tensor, k):
        k = min(k, labels_tensor.size(1))
        return labels_tensor[:, :k].any(dim=1).float().mean().item() * 100

    recall_at1 = recall_at(labels, 1)
    recall_at5 = recall_at(labels, 5)
    recall_at10 = recall_at(labels, 10)
    recall_at50 = recall_at(labels, 50)
    class_recall_at1 = recall_at(class_labels, 1)
    class_recall_at3 = recall_at(class_labels, 3)
    class_recall_at5 = recall_at(class_labels, 5)

    metrics = (
        recall_at1,
        recall_at5,
        recall_at10,
        recall_at50,
        class_recall_at1,
        class_recall_at3,
        class_recall_at5,
    )

    if save_results:
        # Full rankings are intentionally materialized only when explicitly
        # requested. Normal training/validation never needs this large object.
        sorted_indices = torch.argsort(pred_sim, dim=-1, descending=True).cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        sorted_index_names = sorted_index_names[:, :-1]
        return (
            *metrics,
            reference_names,
            modification_texts,
            target_names,
            sorted_index_names,
            reference_bboxes,
            activation_scalars,
        )

    return metrics


def generate_oacirr_val_predictions(
    blip_model,
    relative_val_dataset: OACIRRDataset,
    index_names: List[str],
    index_features,
    txt_processors,
    highlight_inference: bool = False,
    save_memory: bool = False,
    eval_batch_size: int = 32,
):
    """
    Original OACIRR prediction generation.
    """
    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation predictions...")

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=eval_batch_size,
        num_workers=6,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    name_to_feat = dict(zip(index_names, index_features[1]))

    distance = []
    scalar = []

    reference_names = []
    target_names = []
    captions_all = []
    reference_bboxes_all = []

    for batch_data in tqdm(relative_val_loader, ncols=140, ascii=True):
        batch_reference_names = batch_data[0]
        batch_target_names = batch_data[1]
        captions = batch_data[2]

        if len(batch_data) == 5:
            batch_reference_bbox = batch_data[4]
        else:
            batch_reference_bbox = batch_data[3]

        captions = [txt_processors["eval"](caption) for caption in captions]

        with torch.no_grad():
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))

            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(device)

            reference_image_features = reference_image_features.to(device)

            reference_bbox = batch_reference_bbox if highlight_inference else None

            model_output = blip_model.inference(
                reference_image_features,
                feature_curr,
                captions,
                reference_bbox,
            )

            if isinstance(model_output, tuple):
                batch_distance, activation_scalar = model_output
                scalar.append(activation_scalar.cpu())
            else:
                batch_distance = model_output
                scalar.append(torch.zeros(reference_image_features.shape[0], 1))

            distance.append(batch_distance.cpu())

            captions_all += captions
            reference_names.extend(batch_reference_names)
            target_names.extend(batch_target_names)
            reference_bboxes_all.extend(batch_reference_bbox)

    return (
        torch.vstack(distance),
        reference_names,
        target_names,
        captions_all,
        reference_bboxes_all,
        torch.vstack(scalar),
    )


def generate_oacirr_val_predictions_latent(
    blip_model,
    relative_val_dataset: OACIRRDataset,
    index_names: List[str],
    index_features,
    index_raw_embeds,
    txt_processors,
    save_memory: bool = False,
    latent_gallery_chunk_size: int = 1024,
    eval_batch_size: int = 32,
):
    """
    OACIRR prediction generation with AdaFocal retrieval and CORE reranking.

    It uses:
        reference raw visual tokens from index_raw_embeds,
        target retrieval features from index_features,
        target raw visual tokens from index_raw_embeds.
    """
    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation predictions with latent matching...")

    latent_model = blip_model.module if hasattr(blip_model, "module") else blip_model
    if not hasattr(latent_model, "inference_core_matcher_candidates"):
        raise AttributeError(
            "The current model does not implement CORE candidate reranking. "
            "Please use oacir_latent model."
        )

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=eval_batch_size,
        num_workers=6,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    name_to_raw = dict(zip(index_names, index_raw_embeds))

    distance = []
    scalar = []

    reference_names = []
    target_names = []
    captions_all = []
    reference_bboxes_all = []

    num_gallery = len(index_names)

    for batch_data in tqdm(relative_val_loader, ncols=140, ascii=True):
        batch_reference_names = batch_data[0]
        batch_target_names = batch_data[1]
        captions = batch_data[2]

        if len(batch_data) == 5:
            batch_reference_bbox = batch_data[4]
        else:
            batch_reference_bbox = batch_data[3]

        captions = [txt_processors["eval"](caption) for caption in captions]

        with torch.no_grad():
            if len(captions) == 1:
                reference_raw_embeds = itemgetter(*batch_reference_names)(name_to_raw).unsqueeze(0)
            else:
                reference_raw_embeds = torch.stack(itemgetter(*batch_reference_names)(name_to_raw))

            composition_chunks = []
            activation_scalar_for_batch = None

            # CORE reranking needs reference bboxes. We always pass bboxes here.
            reference_bbox = batch_reference_bbox

            for start in range(0, num_gallery, latent_gallery_chunk_size):
                end = min(start + latent_gallery_chunk_size, num_gallery)

                feature_chunk = index_features[start:end]

                composition_chunk = latent_model.inference_composition_from_raw(
                    reference_image_embeds_raw=reference_raw_embeds,
                    target_features=feature_chunk,
                    modification_text=captions,
                    reference_bbox=reference_bbox,
                )
                composition_chunks.append(composition_chunk.detach().cpu())

            global_composition = torch.cat(composition_chunks, dim=1)
            candidate_count = min(
                max(int(latent_model.core_matcher_topk), 1),
                global_composition.size(1),
            )
            candidate_indices = global_composition.topk(
                candidate_count,
                dim=1,
            ).indices
            composition_candidate_logits = torch.gather(
                global_composition,
                dim=1,
                index=candidate_indices,
            )
            reranked_chunks = []
            fusion_scalar_chunks = []
            query_chunk_size = max(
                int(latent_model.core_matcher_query_chunk_size),
                1,
            )
            for query_start in range(0, len(captions), query_chunk_size):
                query_end = min(query_start + query_chunk_size, len(captions))
                query_candidate_indices = candidate_indices[query_start:query_end]
                target_candidate_raw = index_raw_embeds[
                    query_candidate_indices.to(index_raw_embeds.device)
                ]
                rerank_output = latent_model.inference_core_matcher_candidates(
                    reference_image_embeds_raw=reference_raw_embeds[query_start:query_end],
                    target_candidate_embeds_raw=target_candidate_raw,
                    composition_candidate_logits=composition_candidate_logits[query_start:query_end],
                    modification_text=captions[query_start:query_end],
                    reference_bbox=reference_bbox[query_start:query_end],
                    return_parts=True,
                )
                reranked_chunks.append(rerank_output["sim_final"].detach().cpu())
                fusion_scalar_chunks.append(
                    rerank_output["fusion_scalar"].detach().cpu()
                )
                del target_candidate_raw, rerank_output

            reranked_logits = torch.cat(reranked_chunks, dim=0)
            batch_distance = latent_model.merge_topk_ranking(
                global_composition,
                candidate_indices,
                reranked_logits,
            )
            activation_scalar_for_batch = torch.cat(
                fusion_scalar_chunks,
                dim=0,
            )
            distance.append(batch_distance)

            if activation_scalar_for_batch is not None:
                scalar.append(activation_scalar_for_batch.detach().cpu())
            else:
                scalar.append(torch.zeros(reference_raw_embeds.shape[0], 1))

            captions_all += captions
            reference_names.extend(batch_reference_names)
            target_names.extend(batch_target_names)
            reference_bboxes_all.extend(batch_reference_bbox)

    return (
        torch.vstack(distance),
        reference_names,
        target_names,
        captions_all,
        reference_bboxes_all,
        torch.vstack(scalar),
    )


def compute_oacirr_bounding_box_val_metrics(
    relative_val_dataset: OACIRRDataset,
    blip_model,
    index_features,
    index_features_bounding_box,
    index_names: List[str],
    index_names_bounding_box: List[str],
    txt_processors,
    save_results: bool = False,
    save_memory: bool = False,
    eval_batch_size: int = 32,
):
    """
    Compute OACIRR validation metrics with visual bounding-box baseline.
    """
    pred_sim, reference_names, target_names, modification_texts = generate_oacirr_bounding_box_val_predictions(
        blip_model,
        relative_val_dataset,
        index_names_bounding_box,
        index_features,
        index_features_bounding_box,
        txt_processors,
        save_memory,
        eval_batch_size,
    )

    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation metrics...")

    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1)
    sorted_index_names = np.array(index_names)[sorted_indices]

    reference_mask = torch.tensor(
        sorted_index_names
        != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(
        sorted_index_names.shape[0],
        sorted_index_names.shape[1] - 1,
    )

    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1)
    )

    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(len(target_names)).int(),
    )

    class_recall_at1, class_recall_at3, class_recall_at5 = compute_class_recall(
        sorted_index_names,
        target_names,
        relative_val_dataset.name_to_relpath,
    )

    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    metrics = (
        recall_at1,
        recall_at5,
        recall_at10,
        recall_at50,
        class_recall_at1,
        class_recall_at3,
        class_recall_at5,
    )

    if save_results:
        return *metrics, reference_names, modification_texts, target_names, sorted_index_names

    return metrics


def generate_oacirr_bounding_box_val_predictions(
    blip_model,
    relative_val_dataset: OACIRRDataset,
    index_names_bounding_box: List[str],
    index_features,
    index_features_bounding_box,
    txt_processors,
    save_memory: bool = False,
    eval_batch_size: int = 32,
):
    """
    Generate predictions for OACIRR visual bounding-box baseline.
    """
    print(f"Computing OACIRR [{relative_val_dataset.variant}] validation predictions...")

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=eval_batch_size,
        num_workers=6,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    name_to_feat = dict(zip(index_names_bounding_box, index_features_bounding_box[1]))

    distance = []
    reference_names = []
    target_names = []
    captions_all = []

    for batch_data in tqdm(relative_val_loader, ncols=140, ascii=True):
        batch_reference_names = batch_data[0]
        batch_target_names = batch_data[1]
        captions = batch_data[2]
        captions = [txt_processors["eval"](caption) for caption in captions]

        with torch.no_grad():
            if len(captions) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))

            feature_curr = index_features[0]
            if save_memory:
                feature_curr = feature_curr.to(device)

            reference_image_features = reference_image_features.to(device)

            batch_distance = blip_model.inference(
                reference_image_features,
                feature_curr,
                captions,
            )

            distance.append(batch_distance.cpu())
            captions_all += captions
            reference_names.extend(batch_reference_names)
            target_names.extend(batch_target_names)

    return torch.vstack(distance), reference_names, target_names, captions_all


def load_checkpoint_safely(blip_model, weight_path: str):
    """
    Load checkpoint robustly.

    It supports:
        checkpoint[model_class_name]
        checkpoint["Blip2QformerOacirAdaFocal"]
        checkpoint with only one state-dict-like value
        raw state_dict
    """
    checkpoint = torch.load(weight_path, map_location=device)
    class_name = blip_model.__class__.__name__

    state_dict = None

    if isinstance(checkpoint, dict):
        if class_name in checkpoint:
            state_dict = checkpoint[class_name]
        elif "Blip2QformerOacirAdaFocal" in checkpoint:
            state_dict = checkpoint["Blip2QformerOacirAdaFocal"]
        elif "Blip2QformerOacirLatent" in checkpoint:
            state_dict = checkpoint["Blip2QformerOacirLatent"]
        else:
            state_dict_like_values = [
                v for v in checkpoint.values()
                if isinstance(v, dict)
            ]
            if len(state_dict_like_values) == 1:
                state_dict = state_dict_like_values[0]
            else:
                # Maybe checkpoint itself is a raw state dict.
                if all(isinstance(k, str) for k in checkpoint.keys()):
                    state_dict = checkpoint

    if state_dict is None:
        raise KeyError(
            f"Cannot find a valid state dict in checkpoint: {weight_path}. "
            f"Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
        )

    msg = blip_model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys when loading weights: {msg.missing_keys}")
    print(f"Unexpected keys when loading weights: {msg.unexpected_keys}")


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate the model on the OACIRR / Standard CIR Benchmark")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Fashion", "Car", "Product", "Landmark", "CIRR", "FashionIQ"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./Datasets/OACIRR",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--blip-model-name",
        type=str,
        default="oacir_adafocal",
        help="Model registry name",
    )
    parser.add_argument(
        "--blip-model-weight",
        type=str,
        required=True,
        help="Path to the pre-trained model weight",
    )
    parser.add_argument(
        "--vit-backbone",
        type=str,
        default="pretrain",
        help="pretrain for ViT-G, pretrain_vitL for ViT-L",
    )
    parser.add_argument(
        "--target-ratio",
        default=1.25,
        type=float,
        help="TargetPad target ratio",
    )
    parser.add_argument(
        "--transform",
        default="targetpad",
        type=str,
        choices=["squarepad", "targetpad"],
    )

    parser.add_argument(
        "--highlight-inference",
        dest="highlight_inference",
        action="store_true",
        help="Whether to use region highlight strategy during inference",
    )
    parser.add_argument(
        "--text-entity",
        dest="text_entity",
        action="store_true",
        help="Add personalized entity prompt into the modification text",
    )
    parser.add_argument(
        "--bounding-box-width",
        default=0,
        type=int,
        help="Reference image bounding box width",
    )
    parser.add_argument(
        "--bounding-box-color",
        default="red",
        type=str,
        help="Reference image bounding box color",
    )
    parser.add_argument(
        "--bounding-box-crop",
        dest="bounding_box_crop",
        action="store_true",
        help="Crop bounding box region",
    )

    parser.add_argument(
        "--save-results",
        dest="save_results",
        action="store_true",
        help="Whether to save the validation results JSON",
    )
    parser.add_argument(
        "--save-memory",
        dest="save_memory",
        action="store_true",
        help="Save extracted features on CPU",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size for validation query inference and uncached gallery feature extraction.",
    )
    parser.add_argument(
        "--val-feature-cache",
        "--val-feature-cache-template",
        dest="val_feature_cache",
        type=str,
        default=None,
        help=(
            "Path template for precomputed OACIRR val raw embeddings. "
            "Supports {variant} and {variant_lower}, for example "
            "./cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt"
        ),
    )
    parser.add_argument(
        "--val-feature-batch-size",
        type=int,
        default=32,
        help="Batch size for rebuilding gallery features from val raw cache.",
    )

    # CORE reranking arguments.
    parser.add_argument(
        "--use-latent-matching",
        action="store_true",
        help="Use AdaFocal retrieval followed by CORE Top-K reranking",
    )
    parser.add_argument(
        "--core-matcher-temp",
        default=0.07,
        type=float,
        help="Temperature for CORE-Matcher candidate logits",
    )
    parser.add_argument(
        "--core-matcher-lambda-id",
        default=0.5,
        type=float,
        help="Identity weight inside the CORE-Matcher reranking score",
    )
    parser.add_argument(
        "--core-matcher-fusion-weight",
        default=0.02,
        type=float,
        help="Residual CORE-Matcher weight inside AdaFocal Top-K",
    )
    parser.add_argument(
        "--core-matcher-topk",
        default=50,
        type=int,
        help="Number of global AdaFocal candidates reranked by CORE-Matcher",
    )
    parser.add_argument(
        "--core-matcher-cycle-weight",
        default=0.25,
        type=float,
        help="Weight of Matcher-style bidirectional cycle consistency",
    )
    parser.add_argument(
        "--core-matcher-text-weight",
        default=0.25,
        type=float,
        help="Text-composition contribution to target-region discovery",
    )
    parser.add_argument(
        "--core-matcher-bbox-floor",
        default=0.05,
        type=float,
        help="Context floor outside the reference box in CORE aggregation",
    )
    parser.add_argument(
        "--core-matcher-query-chunk-size",
        default=8,
        type=int,
        help="Query chunk size for memory-bounded candidate region matching",
    )
    parser.add_argument(
        "--latent-gallery-chunk-size",
        default=1024,
        type=int,
        help="Gallery chunk size for latent matching inference",
    )
    parser.add_argument(
        "--region-topk",
        default=16,
        type=int,
        help="Minimum target patch count retained by adaptive region selection",
    )
    parser.add_argument(
        "--region-topk-max",
        default=96,
        type=int,
        help="Maximum target patch count retained by adaptive region selection",
    )
    parser.add_argument(
        "--region-area-scale",
        default=1.5,
        type=float,
        help="Scale from reference bbox patch count to target region patch count",
    )
    parser.add_argument(
        "--region-spatial-kernel",
        default=3,
        type=int,
        help="Odd smoothing kernel size for the target response map",
    )
    parser.add_argument(
        "--region-spatial-weight",
        default=0.15,
        type=float,
        help="Distance penalty for spatially coherent target regions",
    )
    parser.add_argument(
        "--region-temperature",
        default=0.07,
        type=float,
        help="Target response temperature used by the model",
    )

    args = parser.parse_args()

    if args.use_latent_matching and args.dataset not in ["Fashion", "Car", "Product", "Landmark"]:
        raise ValueError("--use-latent-matching is currently implemented for OACIRR subsets only.")

    if args.use_latent_matching and (args.bounding_box_width > 0 or args.bounding_box_crop):
        raise ValueError("Latent matching should not be combined with visual bbox drawing/cropping baselines.")

    # ==================== Load Model ====================
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=args.blip_model_name,
        model_type=args.vit_backbone,
        is_eval=False,
        device=device,
    )

    load_checkpoint_safely(blip_model, args.blip_model_weight)

    if hasattr(blip_model, "set_latent_config"):
        blip_model.set_latent_config(
            core_matcher_temp=args.core_matcher_temp,
            core_matcher_lambda_id=args.core_matcher_lambda_id,
            core_matcher_fusion_weight=args.core_matcher_fusion_weight,
            core_matcher_topk=args.core_matcher_topk,
            core_matcher_cycle_weight=args.core_matcher_cycle_weight,
            core_matcher_text_weight=args.core_matcher_text_weight,
            core_matcher_bbox_floor=args.core_matcher_bbox_floor,
            core_matcher_query_chunk_size=args.core_matcher_query_chunk_size,
            region_topk=args.region_topk,
            region_topk_max=args.region_topk_max,
            region_area_scale=args.region_area_scale,
            region_spatial_kernel=args.region_spatial_kernel,
            region_spatial_weight=args.region_spatial_weight,
            region_temperature=args.region_temperature,
        )

    blip_model.eval()

    # ==================== Transforms ====================
    input_dim = 224

    if args.transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print("Square pad preprocess pipeline is used")
    elif args.transform == "targetpad":
        preprocess = targetpad_transform(args.target_ratio, input_dim)
        print(f"Target pad with target_ratio={args.target_ratio} preprocess pipeline is used")
    else:
        raise ValueError("Image preprocess transform should be in ['squarepad', 'targetpad']")

    # ==================== OACIRR Evaluation ====================
    if args.dataset in ["Fashion", "Car", "Product", "Landmark"]:
        relative_val_dataset = OACIRRDataset(
            data_root=args.data_root,
            variant=args.dataset,
            split="val",
            mode="relative",
            preprocess=preprocess,
            highlight_inference=args.highlight_inference or args.use_latent_matching,
            text_entity=args.text_entity,
        )

        classic_val_dataset = OACIRRDataset(
            data_root=args.data_root,
            variant=args.dataset,
            split="val",
            mode="classic",
            preprocess=preprocess,
        )

        classic_val_dataset_bounding_box = None
        if args.bounding_box_width:
            classic_val_dataset_bounding_box = OACIRRDataset(
                data_root=args.data_root,
                variant=args.dataset,
                split="val",
                mode="classic",
                preprocess=preprocess,
                bounding_box_width=args.bounding_box_width,
                bounding_box_color=args.bounding_box_color,
            )
        elif args.bounding_box_crop:
            classic_val_dataset_bounding_box = OACIRRDataset(
                data_root=args.data_root,
                variant=args.dataset,
                split="val",
                mode="classic",
                preprocess=preprocess,
                bounding_box_crop=True,
            )

        if args.use_latent_matching:
            if args.val_feature_cache:
                val_feature_cache = args.val_feature_cache.format(
                    variant=args.dataset,
                    variant_lower=args.dataset.lower(),
                )
                (
                    val_index_features,
                    val_index_raw_embeds,
                    val_index_names,
                ) = extract_index_blip_features_from_raw_cache(
                    classic_val_dataset,
                    blip_model,
                    val_feature_cache,
                    save_memory=args.save_memory,
                    batch_size=args.val_feature_batch_size,
                )
            else:
                val_index_features, val_index_raw_embeds, val_index_names = extract_index_blip_features_with_raw(
                    classic_val_dataset,
                    blip_model,
                    args.save_memory,
                    batch_size=args.val_feature_batch_size,
                )

            start_time = time.perf_counter()
            results = compute_oacirr_val_metrics_latent(
                relative_val_dataset=relative_val_dataset,
                blip_model=blip_model,
                index_features=val_index_features,
                index_raw_embeds=val_index_raw_embeds,
                index_names=val_index_names,
                txt_processors=txt_processors,
                save_results=args.save_results,
                save_memory=args.save_memory,
                latent_gallery_chunk_size=args.latent_gallery_chunk_size,
                eval_batch_size=args.eval_batch_size,
            )
            duration = time.perf_counter() - start_time

        else:
            val_index_features, val_index_names = extract_index_blip_features(
                classic_val_dataset,
                blip_model,
                args.save_memory,
                batch_size=args.eval_batch_size,
            )

            if args.bounding_box_width or args.bounding_box_crop:
                val_index_features_bounding_box, val_index_names_bounding_box = extract_index_blip_features(
                    classic_val_dataset_bounding_box,
                    blip_model,
                    args.save_memory,
                    batch_size=args.eval_batch_size,
                )

                start_time = time.perf_counter()
                results = compute_oacirr_bounding_box_val_metrics(
                    relative_val_dataset,
                    blip_model,
                    val_index_features,
                    val_index_features_bounding_box,
                    val_index_names,
                    val_index_names_bounding_box,
                    txt_processors,
                    args.save_results,
                    args.save_memory,
                    args.eval_batch_size,
                )
                duration = time.perf_counter() - start_time

            else:
                start_time = time.perf_counter()
                results = compute_oacirr_val_metrics(
                    relative_val_dataset,
                    blip_model,
                    val_index_features,
                    val_index_names,
                    txt_processors,
                    highlight_inference=args.highlight_inference,
                    save_results=args.save_results,
                    save_memory=args.save_memory,
                    eval_batch_size=args.eval_batch_size,
                )
                duration = time.perf_counter() - start_time

        if args.save_results:
            (
                recall_at1,
                recall_at5,
                recall_at10,
                recall_at50,
                class_recall_at1,
                class_recall_at3,
                class_recall_at5,
                reference_names,
                modification_texts,
                target_names,
                sorted_index_names,
                reference_bboxes,
                activation_scalars,
            ) = results
        else:
            (
                recall_at1,
                recall_at5,
                recall_at10,
                recall_at50,
                class_recall_at1,
                class_recall_at3,
                class_recall_at5,
            ) = results

        results_dict = {
            "R_ID@1": class_recall_at1,
            "R_ID@3": class_recall_at3,
            "R_ID@5": class_recall_at5,
            "R@1": recall_at1,
            "R@5": recall_at5,
            "R@10": recall_at10,
            "R@50": recall_at50,
            "inference_time": duration,
        }

        print("\n" + "=" * 40)
        print(f"Results for OACIRR [{args.dataset}]:")
        print(json.dumps(results_dict, indent=4))
        print("=" * 40 + "\n")

        if args.save_results:
            base_dir = "/".join(args.blip_model_weight.split("/")[:-2])
            if base_dir == "":
                base_dir = "."

            save_path = Path(base_dir) / "saved_results"
            save_path.mkdir(exist_ok=True, parents=True)

            results_to_save = {
                "reference_names": reference_names,
                "modification_texts": modification_texts,
                "target_names": target_names,
                "sorted_index_names": sorted_index_names.tolist(),
                "reference_bboxes": reference_bboxes,
                "activation_scalars": [
                    s.item() if hasattr(s, "numel") and s.numel() == 1 else (
                        s.numpy().tolist() if hasattr(s, "numpy") else s
                    )
                    for s in activation_scalars
                ],
            }

            if args.use_latent_matching:
                file_name = f"validation_results_latent_{args.dataset.lower()}.json"
            elif args.bounding_box_width or args.bounding_box_crop:
                file_name = f"validation_results_bbox_{args.dataset.lower()}.json"
            else:
                file_name = f"validation_results_{args.dataset.lower()}.json"

            with open(save_path / file_name, "w", encoding="utf-8") as file:
                json.dump(results_to_save, file, indent=4)

            print(f"Results successfully saved to {save_path / file_name}")

    # ==================== CIRR Evaluation ====================
    elif args.dataset == "CIRR":
        relative_val_dataset = CIRRDataset(args.data_root, "val", "relative", preprocess)
        classic_val_dataset = CIRRDataset(args.data_root, "val", "classic", preprocess)

        val_index_features, val_index_names = extract_index_blip_features(
            classic_val_dataset,
            blip_model,
            batch_size=args.eval_batch_size,
        )

        start_time = time.perf_counter()
        results = compute_cirr_val_metrics(
            relative_val_dataset,
            blip_model,
            val_index_features,
            val_index_names,
            txt_processors,
            args.eval_batch_size,
        )
        duration = time.perf_counter() - start_time

        (
            group_recall_at1,
            group_recall_at2,
            group_recall_at3,
            recall_at1,
            recall_at5,
            recall_at10,
            recall_at50,
        ) = results

        results_dict = {
            "group_recall_at1": group_recall_at1,
            "group_recall_at2": group_recall_at2,
            "group_recall_at3": group_recall_at3,
            "recall_at1": recall_at1,
            "recall_at5": recall_at5,
            "recall_at10": recall_at10,
            "recall_at50": recall_at50,
            "mean(R@5+R_s@1)": (group_recall_at1 + recall_at5) / 2,
            "arithmetic_mean": mean(results),
            "harmonic_mean": harmonic_mean(results),
            "geometric_mean": geometric_mean(results),
            "inference_time": duration,
        }

        print("\n" + "=" * 40)
        print("Results for CIRR:")
        print(json.dumps(results_dict, indent=4))
        print("=" * 40 + "\n")

    # ==================== FashionIQ Evaluation ====================
    elif args.dataset == "FashionIQ":
        idx_to_dress_mapping = {
            0: "dress",
            1: "toptee",
            2: "shirt",
        }

        recalls_at10 = []
        recalls_at50 = []
        duration = 0

        for idx, dress_type in idx_to_dress_mapping.items():
            relative_val_dataset = FashionIQDataset(
                args.data_root,
                "val",
                [dress_type],
                "relative",
                preprocess,
            )
            classic_val_dataset = FashionIQDataset(
                args.data_root,
                "val",
                [dress_type],
                "classic",
                preprocess,
            )

            index_features, index_names = extract_index_blip_features(
                classic_val_dataset,
                blip_model,
                args.save_memory,
                batch_size=args.eval_batch_size,
            )

            start_time = time.perf_counter()
            recall_at10, recall_at50 = compute_fiq_val_metrics(
                relative_val_dataset,
                blip_model,
                index_features,
                index_names,
                txt_processors,
                args.save_memory,
                args.eval_batch_size,
            )
            duration += time.perf_counter() - start_time

            recalls_at10.append(recall_at10)
            recalls_at50.append(recall_at50)

            torch.cuda.empty_cache()

        results_dict = {}
        for i in range(len(recalls_at10)):
            results_dict[f"{idx_to_dress_mapping[i]}_recall_at10"] = recalls_at10[i]
            results_dict[f"{idx_to_dress_mapping[i]}_recall_at50"] = recalls_at50[i]

        results_dict.update(
            {
                "average_recall_at10": mean(recalls_at10),
                "average_recall_at50": mean(recalls_at50),
                "average_recall": (mean(recalls_at50) + mean(recalls_at10)) / 2,
                "inference_time": duration,
            }
        )

        print("\n" + "=" * 40)
        print("Results for FashionIQ:")
        print(json.dumps(results_dict, indent=4))
        print("=" * 40 + "\n")

    else:
        raise ValueError("Dataset should be in ['Fashion', 'Car', 'Product', 'Landmark', 'CIRR', 'FashionIQ']")
