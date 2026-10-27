#!/usr/bin/env python3
"""Independent train/evaluate entry. No edits or runtime patches to old code.

Install this file in OACIR/tools/ and the new model in lavis/models/blip2_models/.
  python tools/run_oacir_qformer_identity.py train
  python tools/run_oacir_qformer_identity.py evaluate --checkpoint NEW_MODEL.pt --dataset Product
All gallery images receive BOTH scores before ranking. Metric Top50 is not a
candidate filter. Defaults: pretrained initialization, Union, validation F/C/L,
50 epochs, one GPU, batch 64, text length 32, baseline single-positive CE.
"""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys
import tempfile

NAME = 'oacir_qformer_identity'
DOMAINS = ['Fashion', 'Car', 'Product', 'Landmark']
SETTING_KEYS = ['identity_weight', 'identity_temperature', 'cycle_weight', 'bbox_floor',
                'region_topk', 'region_topk_max', 'region_area_scale',
                'region_spatial_kernel', 'region_spatial_weight', 'region_temperature',
                'query_chunk_size', 'gallery_chunk_size', 'max_txt_len']


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate'])
    p.add_argument('--repo-root')
    p.add_argument('--dataset', choices=DOMAINS + ['Union'], default='Union')
    p.add_argument('--validation-domains', choices=DOMAINS, nargs='+', default=None)
    p.add_argument('--data-root', default='./Datasets/OACIRR')
    p.add_argument('--checkpoint', help='Evaluate ONLY; strict load of this new model, never old latent weights')
    p.add_argument('--gpu', type=int, help='ONE physical GPU; otherwise respect one visible GPU or auto-select')
    p.add_argument('--exclude-gpus', type=int, nargs='+', default=[], help='Exclude known-unhealthy physical GPU indices from auto-selection')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--num-epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--num-workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--train-feature-cache', default=None, help='Default auto-detects matching Union cache; empty string uses images')
    p.add_argument('--val-feature-cache', default='./cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt')
    p.add_argument('--val-feature-batch-size', type=int, default=128)
    p.add_argument('--val-query-batch-size', type=int, default=8)
    p.add_argument('--validation-frequency', type=int, default=1)
    p.add_argument('--save-dir', default='./checkpoints/qformer_identity_from_pretrain')
    p.add_argument('--identity-weight', type=float, default=1.25)
    p.add_argument('--identity-temperature', type=float, default=.07)
    p.add_argument('--cycle-weight', type=float, default=.25)
    p.add_argument('--bbox-floor', type=float, default=.05)
    p.add_argument('--region-topk', type=int, default=16, help='Minimum SPATIAL patches, not gallery TopK')
    p.add_argument('--region-topk-max', type=int, default=96)
    p.add_argument('--region-area-scale', type=float, default=1.5)
    p.add_argument('--region-spatial-kernel', type=int, default=3)
    p.add_argument('--region-spatial-weight', type=float, default=.15)
    p.add_argument('--region-temperature', type=float, default=.07)
    p.add_argument('--query-chunk-size', type=int, default=2)
    p.add_argument('--gallery-chunk-size', type=int, default=32)
    p.add_argument('--max-txt-len', type=int, default=32)
    p.add_argument('--loss-retrieval', type=float, default=1.)
    p.add_argument('--loss-identity', type=float, default=1.)
    p.add_argument('--loss-composition', type=float, default=.5)
    a = p.parse_args()
    if a.mode == 'evaluate' and not a.checkpoint:
        p.error('evaluate requires --checkpoint for the NEW model')
    if a.mode == 'train' and a.checkpoint:
        p.error('Training starts from BLIP pretraining; --checkpoint is evaluate-only')
    if min(a.num_epochs, a.batch_size, a.val_feature_batch_size, a.val_query_batch_size,
           a.validation_frequency, a.query_chunk_size, a.gallery_chunk_size) < 1:
        p.error('Epoch, batch, frequency and chunk counts must be positive')
    if a.num_workers < 0 or a.learning_rate <= 0:
        p.error('num-workers must be nonnegative and learning-rate positive')
    if a.batch_size < 2 and a.mode == 'train':
        p.error('Training needs batch-size >=2 for negatives')
    weights = [a.loss_retrieval, a.loss_identity, a.loss_composition]
    if min(weights) < 0 or a.loss_retrieval <= 0:
        p.error('Loss weights must be nonnegative and retrieval weight positive')
    return a


def find_repo(explicit):
    candidates = [Path(explicit)] if explicit else [Path.cwd(), Path(__file__).resolve().parent.parent]
    for candidate in candidates:
        if (candidate / 'data_utils.py').is_file() and (candidate / 'lavis').is_dir():
            return candidate.resolve()
    raise FileNotFoundError('Run in OACIR repo or set --repo-root')


def setup_runtime(args):
    os.environ['TMPDIR'] = '/tmp'
    tempfile.tempdir = '/tmp'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return
    visible = str(args.gpu) if args.gpu is not None else os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True,
                                timeout=15)
        cards = []
        for line in result.stdout.splitlines():
            index, used, free, util = map(int, line.split(','))
            if index not in args.exclude_gpus and used <= 12000 and free >= 10000 and util <= 70:
                cards.append((-free, util, index))
        if not cards:
            raise RuntimeError('No relatively idle GPU; wait or explicitly choose --gpu INDEX')
        _, _, index = min(cards)
        visible = str(index)
    if not visible.strip() or ',' in visible:
        raise ValueError('Exactly one GPU is permitted')
    if visible.isdigit() and int(visible) in args.exclude_gpus:
        raise ValueError('Selected/visible GPU is excluded; unset CUDA_VISIBLE_DEVICES or choose a healthy GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible
    print('Using ONE visible GPU:', visible, '(no processes are stopped)', flush=True)


def amp(args):
    import torch
    return torch.cuda.amp.autocast() if args.device == 'cuda' else nullcontext()


def read_checkpoint(path):
    import torch
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


def load_model(args):
    from omegaconf import OmegaConf
    from lavis.models import load_preprocess
    module = importlib.import_module('lavis.models.blip2_models.blip2_qformer_' + NAME)
    cls = module.Blip2QformerOacirQformerIdentity
    cfg = OmegaConf.load(cls.default_config_path('pretrain'))
    payload = None
    if args.mode == 'evaluate':
        payload = read_checkpoint(args.checkpoint)
        if not isinstance(payload, dict) or payload.get('model_name') != NAME:
            raise ValueError('Not a checkpoint created by the independent single-stage runner')
        cfg.model.load_pretrained = False
        cfg.model.load_finetuned = False
    model = cls.from_config(cfg.model)
    settings = {name: getattr(args, name) for name in SETTING_KEYS}
    if payload is not None:
        model.load_state_dict(payload['model'], strict=True)
        settings = payload['model_settings']  # Evaluation reproduces saved settings, not CLI defaults.
        # Only memory-bounding settings may change without changing saved scoring.
        settings['query_chunk_size'] = args.query_chunk_size
        settings['gallery_chunk_size'] = args.gallery_chunk_size
        args.max_txt_len = settings['max_txt_len']
        print('Strict NEW-model checkpoint load; epoch:', payload['epoch'], flush=True)
        del payload
    model.set_single_stage_config(**settings)
    _, processors = load_preprocess(cfg.preprocess)
    return model.to('cuda:0' if args.device == 'cuda' else 'cpu'), processors


def strict_collate(items):
    if not items or any(item is None for item in items):
        raise ValueError('Dataset returned a failed sample; refusing to silently drop it')
    from utils import custom_collate_fn
    return custom_collate_fn(items)


def merge_metric_topk(values, indices, new_values, offset, limit):
    """Streaming metric bookkeeping AFTER both scores; never filters scoring."""
    import torch
    new_indices = torch.arange(offset, offset + new_values.shape[1]).expand(len(new_values), -1)
    combined_values = torch.cat([values, new_values.cpu()], dim=1)
    combined_indices = torch.cat([indices, new_indices], dim=1)
    selected = combined_values.topk(min(limit, combined_values.shape[1]), dim=1)
    return selected.values, combined_indices.gather(1, selected.indices)


def metrics_from_top(top, targets, names, paths):
    import torch
    lookup = {name: index for index, name in enumerate(names)}
    target_indices = torch.tensor([lookup[name] for name in targets])[:, None]
    exact = top.eq(target_indices)
    identities = [PurePosixPath(str(paths[name]).replace('\\', '/')).parent.as_posix() for name in names]
    ids = {key: index for index, key in enumerate(dict.fromkeys(identities))}
    gallery_ids = torch.tensor([ids[key] for key in identities])
    identity = gallery_ids[top].eq(gallery_ids[target_indices])
    def recall(hits, k):
        return hits[:, :min(k, hits.shape[1])].any(1).double().mean().item() * 100.
    result = {f'R@{k}': recall(exact, k) for k in [1, 5, 10, 50]}
    result.update({f'R_ID@{k}': recall(identity, k) for k in [1, 3, 5]})
    result['arithmetic_mean'] = sum(result[f'R@{k}'] for k in [1, 5, 10, 50]) / 4.
    return result


def validate(model, relative, classic, processors, args):
    import torch
    from torch.utils.data import DataLoader
    from evaluate import extract_index_blip_features_from_raw_cache, extract_index_blip_features_with_raw
    from tqdm import tqdm
    model.eval()
    cache = args.val_feature_cache.format(variant=relative.variant, variant_lower=relative.variant.lower()) if args.val_feature_cache else ''
    with torch.no_grad(), amp(args):
        if cache:
            packed, raw, names = extract_index_blip_features_from_raw_cache(
                classic, model, cache, save_memory=True, batch_size=args.val_feature_batch_size)
        else:
            packed, raw, names = extract_index_blip_features_with_raw(
                classic, model, save_memory=True, batch_size=args.val_feature_batch_size,
                num_workers=args.num_workers)
        names = list(names)
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError('Gallery must contain >=2 unique names')
        if set(names) != set(classic.name_to_relpath):
            raise ValueError('Gallery extraction skipped/added images; refusing incomplete metrics')
        if packed.shape[1] != model.num_query_token + raw.shape[1] - 1:
            raise ValueError('Unexpected packed Q-Former + identity gallery shape')
        lookup = {name: i for i, name in enumerate(names)}
        loader = DataLoader(relative, batch_size=args.val_query_batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=strict_collate, pin_memory=False)
        top_rows, references, targets = [], [], []
        scored_pairs = 0
        for batch in tqdm(loader, desc=f'FULL gallery [{relative.variant}]', ncols=120):
            refs, tgts, text = list(batch[0]), list(batch[1]), list(batch[2])
            boxes = batch[4] if len(batch) == 5 else batch[3]
            text = [processors['eval'](value) for value in text]
            ref_indices = torch.tensor([lookup[name] for name in refs])
            # Labels are NOT passed into model scoring. Reference IDs are only for exclusion.
            query = model.encode_query_from_raw(raw[ref_indices], text, boxes)
            values = torch.empty(len(refs), 0)
            indices = torch.empty(len(refs), 0, dtype=torch.long)
            for start in range(0, len(names), args.gallery_chunk_size):
                chunk = packed[start:start + args.gallery_chunk_size].to(model._get_device())
                scores = model.score_all_pairs(*query, chunk[:, :model.num_query_token],
                                               chunk[:, model.num_query_token:])['final'].float().cpu()
                if not torch.isfinite(scores).all():
                    raise ValueError('Nonfinite full-gallery scores')
                scored_pairs += scores.numel()
                for row, ref_index in enumerate(ref_indices.tolist()):
                    if start <= ref_index < start + scores.shape[1]:
                        scores[row, ref_index - start] = -torch.inf
                values, indices = merge_metric_topk(values, indices, scores, start, min(50, len(names) - 1))
            top_rows.append(indices)
            references.extend(refs)
            targets.extend(tgts)
        expected_refs = [item['reference'] for item in relative.quadruples]
        expected_targets = [item['target'] for item in relative.quadruples]
        if references != expected_refs or targets != expected_targets:
            raise ValueError('Validation queries were skipped/reordered; refusing incomplete metrics')
        if scored_pairs != len(relative) * len(names):
            raise AssertionError('Not every query/gallery pair was scored')
        result = metrics_from_top(torch.cat(top_rows), targets, names, relative.name_to_relpath)
        result.update(queries=len(targets), gallery_images=len(names), scored_pairs=scored_pairs,
                      all_gallery_scored=True, candidate_filter=False)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    del packed, raw
    gc.collect()
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, epoch, args):
    import torch
    temporary = path.with_suffix('.pt.tmp')
    torch.save({'model_name': NAME, 'epoch': epoch, 'model': model.state_dict(),
                'model_settings': {key: getattr(model, key) for key in SETTING_KEYS},
                'training_config': vars(args)}, temporary)
    os.replace(temporary, path)


def main():
    args = arguments()
    repo = find_repo(args.repo_root)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    setup_runtime(args)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data_utils import OACIRRDataset, OACIRRFeatureDataset, targetpad_transform
    from tqdm import tqdm
    if args.device == 'cuda':
        print('Checking CUDA availability (before model loading)...', flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        print('Initializing CUDA random seed...', flush=True)
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    domains = args.validation_domains or (['Fashion', 'Car', 'Landmark'] if args.dataset == 'Union' else [args.dataset])
    if args.dataset != 'Union' and domains != [args.dataset]:
        raise ValueError('Different validation domains require Union training')
    if args.mode == 'train' and args.train_feature_cache is None:
        args.train_feature_cache = ''
        if args.dataset == 'Union':
            for candidate in ['cache/oacirr_union_train_vitg_targetpad125_raw_fp16.pt',
                              'cache/OACIRR_Union_train_raw_visual.pt']:
                if Path(candidate).is_file():
                    args.train_feature_cache = candidate
                    break
    if not Path(args.data_root).is_dir():
        raise FileNotFoundError('Dataset root missing: ' + str(Path(args.data_root).resolve()))
    if args.mode == 'evaluate' and not Path(args.checkpoint).is_file():
        raise FileNotFoundError('New-model checkpoint missing: ' + str(args.checkpoint))
    if args.mode == 'train' and args.train_feature_cache and not Path(args.train_feature_cache).is_file():
        raise FileNotFoundError('Training raw cache missing: ' + str(args.train_feature_cache))
    for domain in domains:
        cache = args.val_feature_cache.format(variant=domain, variant_lower=domain.lower()) if args.val_feature_cache else ''
        if cache and not Path(cache).is_file():
            raise FileNotFoundError('Validation raw cache missing: ' + cache + '; pass --val-feature-cache to a matching cache or empty string to use images')
    output = Path(args.save_dir) / (args.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    model_path = repo / 'lavis/models/blip2_models/blip2_qformer_oacir_qformer_identity.py'
    manifest = {'model_name': NAME, 'arguments': vars(args), 'validation_domains': domains,
                'architecture': 'Q-Former composition + image/bbox-only identity; full-gallery scoring',
                'identity_objective': 'single-positive CE, NO margin, NO identity multi-positive',
                'text_processing': 'existing blip_caption; Q-Former token embedding, no extra text encoder',
                'composition_bbox_conditioning': False, 'identity_bbox_conditioning': True,
                'source_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [model_path, Path(__file__).resolve(),
                              repo / 'lavis/models/blip2_models/blip2_qformer_oacir_latent.py',
                              repo / 'data_utils.py']},
                'status': 'starting'}
    write_json(output / 'run_manifest.json', manifest)
    print('New output directory:', output, flush=True)
    try:
        model, processors = load_model(args)
        preprocess = targetpad_transform(1.25, 224)
        validation = {}
        for domain in domains:
            validation[domain] = (
                OACIRRDataset(args.data_root, domain, 'val', 'relative', preprocess, highlight_inference=True),
                OACIRRDataset(args.data_root, domain, 'val', 'classic', preprocess))
        manifest['model_settings'] = {key: getattr(model, key) for key in SETTING_KEYS}
        manifest['status'] = 'running'
        write_json(output / 'run_manifest.json', manifest)
        if args.mode == 'evaluate':
            results = {domain: validate(model, *datasets, processors, args)
                       for domain, datasets in validation.items()}
            write_json(output / 'evaluation.json', results)
        else:
            if args.train_feature_cache:
                train = OACIRRFeatureDataset(args.data_root, args.dataset, 'train', preprocess,
                    args.train_feature_cache, highlight_training=True, text_entity=False,
                    bounding_box_width=0, bounding_box_color='red', bounding_box_crop=False)
                if len(train.cache_names) != len(set(train.cache_names)):
                    raise ValueError('Training cache has duplicate names')
                required = {item[key] for item in train.quadruples for key in ['reference', 'target']}
                missing = required - set(train.cache_names)
                if missing:
                    raise ValueError('Training cache does not cover this dataset: ' + str(sorted(missing)[:5]))
            else:
                train = OACIRRDataset(args.data_root, args.dataset, 'train', 'relative', preprocess,
                                      highlight_training=True, text_entity=False)
            loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                num_workers=args.num_workers, collate_fn=strict_collate, pin_memory=False)
            if not len(loader):
                raise ValueError('Training dataset is smaller than one batch')
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate, betas=(.9, .98), eps=1e-7, weight_decay=.05)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                pct_start=min(.3, 1.5 / args.num_epochs), div_factor=100.,
                steps_per_epoch=len(loader), epochs=args.num_epochs)
            scaler = torch.cuda.amp.GradScaler(enabled=args.device == 'cuda')
            weights = {'loss_retrieval': args.loss_retrieval, 'loss_identity': args.loss_identity,
                       'loss_composition': args.loss_composition}
            best = {'identity': -1., 'image': -1., 'primary3': -1.}
            saved = output / 'saved_models'
            saved.mkdir()
            for epoch in range(args.num_epochs):
                model.train()
                totals = {key: 0. for key in weights}
                count = 0
                progress = tqdm(loader, desc=f'epoch {epoch}', ncols=120)
                for batch in progress:
                    optimizer.zero_grad(set_to_none=True)
                    key = 'image_embeds_raw' if args.train_feature_cache else 'image'
                    samples = {'reference_' + key: batch[0].to(model._get_device()),
                               'target_' + key: batch[1].to(model._get_device()),
                               'modification_text': [processors['eval'](text) for text in batch[4]],
                               'reference_bbox': batch[5]}
                    with amp(args):
                        losses = model(samples)
                        loss = sum(weights[name] * value for name, value in losses.items())
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    number = len(batch[0])
                    count += number
                    for name, value in losses.items():
                        totals[name] += value.detach().float().item() * number
                    progress.set_postfix(loss=float(loss.detach()))
                append_csv(output / 'train_metrics.csv', {'epoch': epoch,
                    **{name: value / count for name, value in totals.items()}})
                if epoch % args.validation_frequency == 0 or epoch == args.num_epochs - 1:
                    results = {}
                    for domain, datasets in validation.items():
                        results[domain] = validate(model, *datasets, processors, args)
                        append_csv(output / (domain.lower() + '_validation_metrics.csv'),
                                   {'epoch': epoch, **results[domain]})
                        if args.device == 'cuda':
                            torch.cuda.empty_cache()
                    means = {
                        'identity': sum(r['R_ID@1'] for r in results.values()) / len(results),
                        'image': sum(r['arithmetic_mean'] for r in results.values()) / len(results),
                        'primary3': sum((r['R_ID@1'] + r['R@1'] + r['R@5']) / 3.
                                        for r in results.values()) / len(results)}
                    for criterion, value in means.items():
                        if value > best[criterion]:
                            best[criterion] = value
                            save_checkpoint(saved / ('best_' + criterion + '.pt'), model, epoch, args)
                    write_json(output / 'best_metrics.json', {'epoch_last_validated': epoch, **best})
        manifest['status'] = 'complete'
        write_json(output / 'run_manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        write_json(output / 'run_manifest.json', manifest)
        raise
    print('Results:', output, flush=True)


if __name__ == '__main__':
    main()
