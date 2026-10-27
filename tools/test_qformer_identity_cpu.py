"""CPU contract tests. Real CORE/matcher, tiny image/Q-Former substitutes.

No pretrained weights, GPU inference or real-data training. Does not modify
old source files. Installed: python tools/test_qformer_identity_cpu.py --repo-root .
"""
import argparse
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This test is always CPU-only.
import torch
from torch import nn
from torch.nn import functional as F


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Tokens:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.attention_mask = input_ids.ne(0).long()
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class Tokenizer:
    def __len__(self):
        return 64
    def __call__(self, texts, max_length, **kwargs):
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [1] + [2 + sum(map(ord, word)) % 60 for word in text.split()] + [2]
            tokens = tokens[:max_length]
            ids[row, :len(tokens)] = torch.tensor(tokens)
        return Tokens(ids)


class TinyBert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.position_embeddings = nn.Embedding(512, width)
        self.embeddings.word_embeddings = nn.Embedding(64, width)
        self.image = nn.Linear(width, width)
        self.mix = nn.Linear(width, width)
        self.calls = []
    def forward(self, input_ids=None, query_embeds=None, encoder_hidden_states=None, **kwargs):
        assert 'attention_bias' not in kwargs, 'CAAM bias unexpectedly supplied'
        assert query_embeds is not None
        self.calls.append((input_ids is not None, query_embeds.shape[1]))
        visual = self.image(encoder_hidden_states.mean(1))[:, None]
        query = query_embeds + visual
        if input_ids is not None:
            text = self.embeddings.word_embeddings(input_ids)
            text_mask = input_ids.ne(0).unsqueeze(-1)
            context = (text * text_mask).sum(1, keepdim=True) / text_mask.sum(1, keepdim=True).clamp_min(1)
            # Tiny stand-in for contextual self-attention: CLS must see other words.
            text = self.mix(text + visual + context)
            query = query + text.mean(1, keepdim=True)
            output = torch.cat([query, text], dim=1)
        else:
            output = query
        return types.SimpleNamespace(last_hidden_state=output)


class TinyQFormer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=width)
        self.bert = TinyBert(width)
    def resize_token_embeddings(self, size):
        pass


class Vision(nn.Module):
    num_features = 8
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, images):
        return images * self.scale


def real_qformer_test(repo):
    """Installed Q-Former implementation, small random config; no pretrained ViT."""
    sys.path.insert(0, str(repo))
    from transformers import BertConfig
    from lavis.models.blip2_models.Qformer import BertLMHeadModel
    from lavis.models.blip2_models.blip2_qformer_oacir_qformer_identity import (
        Blip2QformerOacirQformerIdentity as Model,
    )
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    def qformer(self, count, width, frequency):
        config = BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, max_position_embeddings=128,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        config.encoder_width = width
        config.add_cross_attention = True
        config.cross_attention_freq = frequency
        config.query_length = count
        return BertLMHeadModel(config), nn.Parameter(torch.randn(1, count, config.hidden_size) * .02)
    with patch.object(Model, 'init_tokenizer', lambda self: Tokenizer()), \
         patch.object(Model, 'init_vision_encoder', lambda self, *a: (Vision(), nn.LayerNorm(8))), \
         patch.object(Model, 'init_Qformer', qformer):
        model = Model(num_query_token=3, embed_dim=8)
    model.set_single_stage_config(query_chunk_size=1, gallery_chunk_size=1,
        max_txt_len=12, region_topk=2, region_topk_max=8)
    samples = {'reference_image_embeds_raw': torch.randn(2, 17, 8),
               'target_image_embeds_raw': torch.randn(2, 17, 8),
               'modification_text': ['make it red', 'change background'],
               'reference_bbox': [[0, 0, 112, 224], [56, 56, 168, 168]]}
    model.train()
    losses = model(samples)
    total = losses['loss_retrieval'] + losses['loss_identity'] + .5 * losses['loss_composition']
    total.backward()
    for parameter in [model.query_tokens, model.text_proj.weight, model.vision_proj.weight,
                      model.instance_proj.weight, model.core_reference_encoder.activation_head.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        qa = model.encode_query_from_raw(samples['reference_image_embeds_raw'],
            samples['modification_text'], samples['reference_bbox'])
        qb = model.encode_query_from_raw(samples['reference_image_embeds_raw'],
            ['different text'] * 2, samples['reference_bbox'])
        bank = model.encode_gallery_from_raw(samples['target_image_embeds_raw'])
        sa = model.score_all_pairs(*qa, *bank)
        sb = model.score_all_pairs(*qb, *bank)
        torch.testing.assert_close(sa['identity'], sb['identity'], rtol=0, atol=0)
        assert not torch.allclose(sa['composition'], sb['composition'])
    print('REAL_QFORMER_CPU_PASS: installed Q-Former small random config, real CORE, '
          'three-loss backward, text-dependent composition and text-free identity. '
          'NO pretrained ViT/checkpoint/GPU test.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--real-qformer', action='store_true', help='Use actual installed Q-Former, tiny random configuration')
    args = parser.parse_args()
    if args.real_qformer:
        if not args.repo_root:
            parser.error('--real-qformer requires --repo-root')
        real_qformer_test(args.repo_root.resolve())
        return
    here = Path(__file__).resolve().parent
    if args.repo_root:
        repo = args.repo_root.resolve()
        source = repo / 'lavis/models/blip2_models'
        new_path = source / 'blip2_qformer_oacir_qformer_identity.py'
        runner_path = repo / 'tools/run_oacir_qformer_identity.py'
    else:
        repo = here.parents[1]
        source = repo / 'work/identity_patch/lavis/models/blip2_models'
        new_path = here / 'blip2_qformer_oacir_qformer_identity.py'
        runner_path = here / 'run_oacir_qformer_identity.py'
    protected = [source / 'blip2_qformer_oacir_latent.py', source / 'oacir_latent_matching.py']
    if args.repo_root:
        protected += [repo / 'train.py', repo / 'evaluate.py', repo / 'lavis/models/__init__.py']
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    for name in ['lavis', 'lavis.common', 'lavis.models', 'lavis.models.blip2_models']:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    class Registry:
        models = {}
        def register_model(self, name):
            def decorate(cls):
                assert name not in self.models
                self.models[name] = cls
                return cls
            return decorate
    registration = types.ModuleType('lavis.common.registry')
    registration.registry = Registry()
    sys.modules[registration.__name__] = registration
    base_module = types.ModuleType('lavis.models.blip2_models.blip2')
    base_module.Blip2Base = nn.Module
    base_module.disabled_train = lambda self, mode=True: self
    sys.modules[base_module.__name__] = base_module
    prefix = 'lavis.models.blip2_models.'
    matcher = load(prefix + 'oacir_latent_matching', protected[1])
    base = load(prefix + 'blip2_qformer_oacir_latent', protected[0])
    new = load(prefix + 'blip2_qformer_oacir_qformer_identity', new_path)
    runner = load('test_single_stage_runner', runner_path)
    cls = new.Blip2QformerOacirQformerIdentity
    torch.set_num_threads(1)
    torch.manual_seed(2026)

    # Exercise the ACTUAL NEW constructor; substitute only large pretrained encoders.
    with patch.object(cls, 'init_tokenizer', lambda self: Tokenizer(), create=True), \
         patch.object(cls, 'init_vision_encoder', lambda self, *a: (Vision(), nn.LayerNorm(8)), create=True), \
         patch.object(cls, 'init_Qformer', lambda self, n, *a: (TinyQFormer(8), nn.Parameter(torch.randn(1, n, 8))), create=True):
        model = cls(num_query_token=3, embed_dim=8)
        second = cls(num_query_token=3, embed_dim=8)
    model.set_single_stage_config(query_chunk_size=2, gallery_chunk_size=3,
        region_topk=2, region_topk_max=8, max_txt_len=12)
    second.set_single_stage_config(query_chunk_size=2, gallery_chunk_size=3,
        region_topk=2, region_topk_max=8, max_txt_len=12)
    assert model.core_reference_encoder.num_regions == 16
    forbidden = ['crm', 'probe', 'adaptive_fusion', 'caam', 'film']
    assert not any(any(term in key.lower() for term in forbidden) for key in model.state_dict())
    assert not hasattr(model, 'inference_core_matcher_candidates')
    assert not hasattr(model, '_build_candidate_indices')
    assert not hasattr(model, 'merge_topk_ranking')
    assert registration.registry.models['oacir_latent'] is base.Blip2QformerOacirLatent
    assert registration.registry.models['oacir_qformer_identity'] is cls
    assert all(not p.requires_grad for p in model.visual_encoder.parameters())
    assert 'modification_text' not in inspect.signature(model.encode_identity_reference).parameters
    assert 'composed_features' not in inspect.signature(new.identity_region_scores).parameters
    print('PASS actual constructor, new registration, no removed modules/candidate API, frozen ViT')

    refs = torch.randn(3, 17, 8)
    gallery = torch.randn(7, 17, 8)
    boxes = [[0, 0, 112, 224], [56, 56, 168, 168], [0, 112, 224, 224]]
    text_a = ['red car', 'change shoes', 'white bag']
    text_b = ['blue vehicle', 'different background', 'black object']
    model.eval()
    a = model.encode_query_from_raw(refs, text_a, boxes)
    b = model.encode_query_from_raw(refs, text_b, boxes)
    target = model.encode_gallery_from_raw(gallery)
    scores_a = model.score_all_pairs(*a, *target)
    scores_b = model.score_all_pairs(*b, *target)
    changed_boxes = [[0, 0, 224, 224]] * 3
    changed_query = model.encode_query_from_raw(refs, text_a, changed_boxes)
    torch.testing.assert_close(a[0], changed_query[0], rtol=0, atol=0)
    assert scores_a['final'].shape == (3, 7)
    torch.testing.assert_close(scores_a['identity'], scores_b['identity'], rtol=0, atol=0)
    assert not torch.allclose(scores_a['composition'], scores_b['composition'])
    no_text_grad = torch.autograd.grad(scores_a['identity'].sum(),
        tuple(model.text_proj.parameters()) + tuple(model.Qformer.parameters()), allow_unused=True, retain_graph=True)
    assert all(g is None for g in no_text_grad)
    torch.testing.assert_close(scores_a['final'], scores_a['composition'] + model.identity_weight * scores_a['identity'])
    assert {has_text for has_text, _ in model.Qformer.bert.calls} == {False, True}
    assert all(count == 3 for _, count in model.Qformer.bert.calls)
    print('PASS text-dependent composition, exactly text-invariant identity scores, no text/Q-Former identity gradient')

    legacy = matcher.compute_bidirectional_region_scores(a[1], target[1][None].expand(3, -1, -1, -1),
        torch.zeros(3, 8), a[2], spatial_kernel_size=model.region_spatial_kernel,
        spatial_weight=model.region_spatial_weight, region_temperature=model.region_temperature,
        cycle_weight=model.cycle_weight, text_weight=0., return_heatmap=True)
    actual, attention = new.identity_region_scores(a[1], target[1], a[2],
        model.cycle_weight, model.region_spatial_kernel, model.region_spatial_weight,
        model.region_temperature, return_attention=True)
    torch.testing.assert_close(actual, legacy['score_id'], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(attention, legacy['target_attention'], rtol=1e-5, atol=1e-6)
    print('PASS dedicated no-text matcher matches previous M_id scores and region attention')

    # Train-mode checkpointed blocks vs one unchunked computation, including gradients.
    def compute(chunk_q, chunk_g):
        model.query_chunk_size, model.gallery_chunk_size = chunk_q, chunk_g
        model.train()
        query = model.encode_query_from_raw(refs, text_a, boxes)
        bank = model.encode_gallery_from_raw(gallery)
        scores = model.score_all_pairs(*query, *bank)
        params = tuple(p for p in model.parameters() if p.requires_grad)
        gradients = torch.autograd.grad(scores['final'].square().mean(), params, allow_unused=True)
        return {k: v.detach() for k, v in scores.items()}, gradients
    chunked, chunked_grad = compute(2, 3)
    full, full_grad = compute(3, 7)
    for key in full:
        torch.testing.assert_close(chunked[key], full[key], rtol=1e-5, atol=2e-5)
    for x, y in zip(chunked_grad, full_grad):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, rtol=3e-4, atol=3e-5)
            assert torch.isfinite(x).all()
    print('PASS full-pair chunked/dense score and backward equivalence (training checkpointing enabled)')

    model.query_chunk_size, model.gallery_chunk_size = 2, 3
    samples = {'reference_image_embeds_raw': refs, 'target_image_embeds_raw': gallery[:3],
               'modification_text': text_a, 'reference_bbox': boxes}
    model.train()
    model.zero_grad(set_to_none=True)
    losses = model(samples)
    assert set(losses) == {'loss_retrieval', 'loss_identity', 'loss_composition'}
    assert all(torch.isfinite(v) for v in losses.values())
    total = losses['loss_retrieval'] + losses['loss_identity'] + .5 * losses['loss_composition']
    total.backward()
    for parameter in [model.query_tokens, model.text_proj.weight, model.vision_proj.weight,
                      model.instance_proj.weight, model.core_reference_encoder.activation_head.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.visual_encoder.scale.grad is None
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        cached = model(samples)
        images = model({'reference_image': refs, 'target_image': gallery[:3],
                        'modification_text': text_a, 'reference_bbox': boxes})
    for key in cached:
        torch.testing.assert_close(cached[key], images[key], rtol=0, atol=0)
    print('PASS complete three-loss forward/backward + toy optimizer step, cached/image path agreement')

    for bad_boxes in [None, [None] * 3, [[0, 0, 0, 0]] * 3]:
        try:
            model.encode_identity_reference(refs, bad_boxes)
        except ValueError:
            pass
        else:
            raise AssertionError('Bad/missing bbox silently accepted')
    try:
        model.set_single_stage_config(core_matcher_topk=50)
    except ValueError:
        pass
    else:
        raise AssertionError('Gallery candidate setting accepted')

    with tempfile.TemporaryDirectory(prefix='oacir_single_stage_') as directory:
        path = Path(directory) / 'model.pt'
        runner.save_checkpoint(path, model, 7, types.SimpleNamespace(test=True))
        payload = runner.read_checkpoint(path)
        assert payload['model_name'] == 'oacir_qformer_identity' and payload['epoch'] == 7
        second.load_state_dict(payload['model'], strict=True)
        second.set_single_stage_config(**payload['model_settings'])
        second.eval()
        for key, value in model(samples).items():
            torch.testing.assert_close(value, second(samples)[key], rtol=0, atol=0)
    print('PASS input guards, strict new-checkpoint roundtrip and saved model settings')

    # Actual streaming evaluator, with toy dataset/model/cache only. Target is
    # OUTSIDE composition Top50, but pure identity promotes it to global Top1.
    names = ['g' + str(i) for i in range(65)]
    packed = torch.zeros(65, 6, 1)
    packed[:, 0, 0] = torch.arange(65, 0, -1).float()
    packed[-1, 2, 0] = 100.
    raw = torch.zeros(65, 5, 1)
    paths = {name: 'Product/id_' + str(i) + '/' + name + '.jpg' for i, name in enumerate(names)}
    class Relative:
        variant = 'Product'
        quadruples = [{'reference': names[0], 'target': names[-1]}]
        name_to_relpath = paths
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return names[0], names[-1], 'modify object', [0., 0., 224., 224.]
    class ToyModel(nn.Module):
        num_query_token = 2
        def __init__(self):
            super().__init__()
            self.seen = 0
        def _get_device(self):
            return torch.device('cpu')
        def encode_query_from_raw(self, raw, text, boxes):
            return (None, None, None)
        def score_all_pairs(self, q, r, size, gallery_features, gallery_patches):
            self.seen += len(gallery_features)
            return {'final': (gallery_features[:, 0, 0] + gallery_patches[:, 0, 0])[None]}
    fake_evaluate = types.ModuleType('evaluate')
    fake_evaluate.extract_index_blip_features_from_raw_cache = lambda *a, **kw: (packed, raw, names)
    fake_evaluate.extract_index_blip_features_with_raw = lambda *a, **kw: (packed, raw, names)
    fake_utils = types.ModuleType('utils')
    fake_utils.custom_collate_fn = lambda items: tuple(list(values) for values in zip(*items))
    test_args = types.SimpleNamespace(val_feature_cache='synthetic', val_feature_batch_size=2,
        val_query_batch_size=1, num_workers=0, gallery_chunk_size=7, device='cpu')
    toy = ToyModel()
    with patch.dict(sys.modules, {'evaluate': fake_evaluate, 'utils': fake_utils}):
        result = runner.validate(toy, Relative(), types.SimpleNamespace(name_to_relpath=paths),
                                 {'eval': lambda s: s}, test_args)
    assert names.index(names[-1]) not in packed[:, 0, 0].topk(50).indices.tolist()
    assert toy.seen == 65 and result['scored_pairs'] == 65
    assert result['R@1'] == result['R_ID@1'] == 100.
    top = torch.tensor([[1, 2]])
    paths['g1'] = paths['g2'] = 'Product/shared/any.jpg'
    metrics = runner.metrics_from_top(top, ['g2'], names, paths)
    assert metrics['R@1'] == 0. and metrics['R_ID@1'] == 100. and metrics['R@50'] == 100.
    print('PASS actual evaluator: all gallery pairs scored, target outside semantic Top50 reaches Top1; exact/ID metrics')

    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    print('ALL_CPU_TESTS_PASS (synthetic encoders/data only; old source hashes unchanged)')


if __name__ == '__main__':
    main()
