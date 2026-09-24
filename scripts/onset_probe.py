#!/usr/bin/env python3
"""Position-matched first-onset probing: prepare, extract, then train.

Indices in the manifest refer to the original stored token array. LLaVA's
Transformers 4.37 inputs_embeds generation prepends a synthetic BOS to returned
sequences; it is NOT fed back after the prompt. Extraction verifies this behavior
and maps raw token index k to expanded_prompt_length + k - 1.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DISTANCES = [1, 2, 4, 8]
LAYERS = [16, 32]  # One-based decoder block outputs, BEFORE final RMSNorm.


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(path):
    with open(path) as stream:
        records = [json.loads(line) for line in stream]
    assert len({x['image_id'] for x in records}) == len(records), 'Duplicate images'
    return records


def valid_token(record, index):
    tok = record['tokenization']
    return (0 <= index < len(tok['token_ids'])
            and not tok['special_tokens_mask'][index]
            and tok['char_offsets'][index] is not None)


def prepare(args):
    records = read_records(args.input)
    groups = {0: [], 1: []}
    excluded = 0
    for record in records:
        labels = record['hallucination_start_labels']
        assert len(labels) == len(record['tokenization']['token_ids'])
        onset = next((i for i, v in enumerate(labels) if v), None)
        if onset is not None and not all(valid_token(record, onset-d) for d in DISTANCES):
            excluded += 1
            continue
        groups[int(onset is not None)].append((record, onset))
    rng = random.Random(args.split_seed)
    splits = {name: {0: [], 1: []} for name in ['train', 'val', 'test']}
    for label, group in groups.items():
        rng.shuffle(group)
        a, b = int(len(group)*0.7), int(len(group)*0.85)
        for name, chunk in zip(splits, [group[:a], group[a:b], group[b:]]):
            splits[name][label] = chunk
    samples, stats = [], {}
    for split, groups_in_split in splits.items():
        negatives = [r for r, _ in groups_in_split[0]]
        positives = sorted(groups_in_split[1], key=lambda item: -item[1])
        count = 0
        # Most constrained (latest) onset first; choose nearest caption length.
        for positive, onset in positives:
            candidates = [i for i, n in enumerate(negatives)
                          if valid_token(n, onset)
                          and all(valid_token(n, onset-d) for d in DISTANCES)]
            if not candidates:
                continue
            idx = min(candidates, key=lambda i: abs(
                len(negatives[i]['tokenization']['token_ids'])
                - len(positive['tokenization']['token_ids'])))
            negative = negatives.pop(idx)
            pair_id = f'{split}-{count:05d}'
            for label, record in [(1, positive), (0, negative)]:
                samples.append(dict(image_id=record['image_id'], label=label,
                                    split=split, pair_id=pair_id, onset_index=onset,
                                    token_indices=[onset-d for d in DISTANCES]))
            count += 1
        stats[split] = dict(pairs=count, samples=2*count,
                            unmatched_positive=len(positives)-count,
                            unused_negative=len(negatives))
    manifest = dict(input=str(Path(args.input).resolve()), input_sha256=digest(args.input),
                    split_seed=args.split_seed, distances=DISTANCES, layers=LAYERS,
                    layer_definition='one-based decoder block output before final RMSNorm',
                    excluded_early_or_special=excluded, stats=stats, samples=samples)
    save_json(args.output / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'samples'}, indent=2), flush=True)


def extract(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    from model.LLaVA import LLaVA
    from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                                 DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX)
    from llava.mm_utils import tokenizer_image_token

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: execute outside the sandbox on a GPU node.')
    torch.set_num_threads(4)
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert digest(manifest['input']) == manifest['input_sha256']
    records = {r['image_id']: r for r in read_records(manifest['input'])}
    adapter = LLaVA(SimpleNamespace(model_path=args.model_path, device='cuda',
                                    temperature=0, num_beams=1, max_length=128,
                                    top_p=None, top_k=None))
    model = adapter.model.eval()
    assert model.config.num_hidden_layers == 32
    samples = manifest['samples']
    feature_path = args.output / 'features.npy'
    shape = (len(samples), len(LAYERS), len(DISTANCES), model.config.hidden_size)
    features = np.lib.format.open_memmap(str(feature_path)+'.partial', mode='w+',
                                         dtype=np.float16, shape=shape)
    capture = {}
    positions = []
    handles = []
    for layer in LAYERS:
        def hook(module, inputs, output, layer=layer):
            capture[layer] = output[0][0, positions, :].detach().float().cpu()
        handles.append(model.get_model().layers[layer-1].register_forward_hook(hook))
    validation = []
    try:
        with torch.inference_mode():
            for row, sample in enumerate(samples):
                record = records[sample['image_id']]
                raw = record['tokenization']['token_ids']
                assert raw[0] == adapter.tokenizer.bos_token_id
                assert all(valid_token(record, k) for k in sample['token_indices'])
                adapter.refresh_chat()
                image_token = DEFAULT_IMAGE_TOKEN
                if model.config.mm_use_im_start_end:
                    image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
                adapter.conv.append_message(adapter.conv.roles[0], image_token+'\n'+record['prompt'])
                adapter.conv.append_message(adapter.conv.roles[1], None)
                prompt = tokenizer_image_token(adapter.conv.get_prompt(), adapter.tokenizer,
                                               IMAGE_TOKEN_INDEX, return_tensors='pt')[None].cuda()
                image = adapter._image_tensor(str(Path(args.coco_root) / record['image_file']))
                _, _, _, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
                    prompt, None, None, None, None, image)
                plen = embeds.shape[1]
                # Confirm the stored leading BOS is generation bookkeeping.
                if row == 0:
                    positions[:] = [-1]
                    generated = model.generate(prompt, images=image, do_sample=False,
                                               max_new_tokens=1, return_dict_in_generate=True,
                                               output_scores=True, use_cache=True)
                    ids = generated.sequences[0].tolist()
                    assert len(ids) == len(generated.scores)+1 and ids[0] == raw[0], ids
                    assert ids[1] == raw[1], 'Prompt/image replay does not reproduce first token'
                last = max(sample['token_indices'])
                response = torch.tensor(raw[1:last+1], device='cuda')[None]
                full = torch.cat([embeds, model.get_model().embed_tokens(response)], dim=1)
                positions[:] = [plen+k-1 for k in sample['token_indices']]
                model(inputs_embeds=full, use_cache=False, return_dict=True)
                saved = {layer: capture[layer].clone() for layer in LAYERS}
                features[row] = torch.stack([saved[l] for l in LAYERS]).half().numpy()
                if row < args.validate_samples:
                    # Forced KV-cache replay of the exact stored prefix. Compare
                    # the same pre-onset positions, never feed onset to the probe.
                    positions[:] = [-1]
                    out = model(inputs_embeds=embeds, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    checks = []
                    for k in range(1, last+1):
                        out = model(input_ids=torch.tensor([[raw[k]]], device='cuda'),
                                    past_key_values=past, use_cache=True, return_dict=True)
                        past = out.past_key_values
                        if k in sample['token_indices']:
                            d_idx = sample['token_indices'].index(k)
                            for layer in LAYERS:
                                a, b = saved[layer][d_idx], capture[layer][0]
                                cosine = float(F.cosine_similarity(a[None], b[None]))
                                relative_l2 = float((a-b).norm()/a.norm().clamp_min(1e-8))
                                checks.append(dict(layer=layer, token_index=k,
                                                   cosine=cosine, relative_l2=relative_l2,
                                                   max_abs=float((a-b).abs().max())))
                                if cosine < 0.999 or relative_l2 > 0.05:
                                    raise RuntimeError(f'Hidden-state replay mismatch: {checks[-1]}')
                    validation.append(dict(image_id=sample['image_id'], checks=checks))
                    del out, past
                if row % 50 == 0 or row+1 == len(samples):
                    features.flush()
                    print(f'Extracted {row+1}/{len(samples)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    features.flush()
    del features
    Path(str(feature_path)+'.partial').replace(feature_path)
    save_json(args.output / 'extraction.json', dict(
        manifest_sha256=digest(manifest_path), model_path=args.model_path,
        shape=shape, dtype='float16', synthetic_bos_removed=True,
        torch_version=torch.__version__, validation=validation))


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                accuracy=float(accuracy_score(y, p >= .5)),
                f1=float(f1_score(y, p >= .5, zero_division=0)))


def train(args):
    import numpy as np
    import torch
    from torch import nn
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(4)
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    extraction = json.loads((args.output/'extraction.json').read_text())
    assert extraction['manifest_sha256'] == digest(manifest_path)
    features = np.load(args.output/'features.npy', mmap_mode='r')
    samples = manifest['samples']
    y = np.array([s['label'] for s in samples])
    indices = {name: np.array([i for i, s in enumerate(samples) if s['split'] == name])
               for name in ['train', 'val', 'test']}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for li, layer in enumerate(LAYERS):
        for di, distance in enumerate(DISTANCES):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            net = nn.Sequential(nn.LayerNorm(features.shape[-1]),
                                nn.Linear(features.shape[-1], 1024), nn.GELU(), nn.Dropout(.2),
                                nn.Linear(1024, 256), nn.GELU(), nn.Dropout(.2),
                                nn.Linear(256, 64), nn.GELU(), nn.Dropout(.2),
                                nn.Linear(64, 1)).to(device)
            x = {s: torch.tensor(np.array(features[ix, li, di]), dtype=torch.float32, device=device)
                 for s, ix in indices.items()}
            labels = {s: torch.tensor(y[ix], dtype=torch.float32, device=device)
                      for s, ix in indices.items()}
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
            best, stale, history = -1., 0, []
            for epoch in range(1, args.epochs+1):
                net.train()
                losses = []
                for batch in torch.randperm(len(x['train']), device=device).split(64):
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        net(x['train'][batch]).flatten(), labels['train'][batch])
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                net.eval()
                with torch.no_grad():
                    vp = net(x['val']).flatten().sigmoid().cpu().numpy()
                score = roc_auc_score(y[indices['val']], vp)
                history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_auroc=float(score)))
                if score > best:
                    best, stale, best_epoch = score, 0, epoch
                    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                if stale >= 5:
                    break
            net.load_state_dict(state)
            net.eval()
            with torch.no_grad():
                probs = net(x['test']).flatten().sigmoid().cpu().numpy()
            test_y = y[indices['test']]
            result = dict(layer=layer, distance=distance, seed=args.seed,
                          best_epoch=best_epoch, val_auroc=float(best), **metrics(test_y, probs))
            # Resample matched image pairs to preserve matching and balance.
            pair_ids = [samples[i]['pair_id'] for i in indices['test']]
            pairs = list(dict.fromkeys(pair_ids))
            pair_ix = np.array([[i for i, p in enumerate(pair_ids) if p == pair] for pair in pairs])
            rng = np.random.default_rng(2026)
            boot = {k: [] for k in ['auroc', 'auprc', 'accuracy', 'f1']}
            for _ in range(args.bootstrap):
                ix = pair_ix[rng.integers(len(pairs), size=len(pairs))].reshape(-1)
                for key, value in metrics(test_y[ix], probs[ix]).items():
                    boot[key].append(value)
            result['ci95'] = {k: np.quantile(v, [.025, .975]).tolist() for k, v in boot.items()}
            name = f'layer{layer}_d{distance}_seed{args.seed}'
            torch.save(dict(state_dict=state, layer=layer, distance=distance, seed=args.seed,
                            manifest_sha256=digest(manifest_path)), args.output/f'{name}.pt')
            save_json(args.output/f'{name}.json', dict(result=result, history=history,
                predictions=[dict(image_id=samples[i]['image_id'], pair_id=samples[i]['pair_id'],
                                  label=int(y[i]), probability=float(p))
                             for i, p in zip(indices['test'], probs)]))
            results.append(result)
            print(json.dumps(result), flush=True)
    save_json(args.output/'results.json', dict(seed=args.seed, config=dict(
        architecture=f'LayerNorm -> {features.shape[-1]} -> 1024 -> 256 -> 64 -> 1; GELU/dropout=0.2',
        lr=1e-4, weight_decay=1e-4, batch_size=64, max_epochs=args.epochs,
        patience=5, bootstrap=args.bootstrap, ci_unit='matched image pair'), results=results))
    fig, ax = plt.subplots(figsize=(6, 4))
    for layer in LAYERS:
        rs = [r for r in results if r['layer'] == layer]
        means = np.array([r['auroc'] for r in rs])
        bounds = np.array([r['ci95']['auroc'] for r in rs])
        ax.errorbar(DISTANCES, means, yerr=np.maximum(0, np.stack([means-bounds[:, 0], bounds[:, 1]-means])),
                    marker='o', capsize=3, label=f'Block {layer}')
    ax.axhline(.5, color='gray', linestyle='--', label='Chance')
    ax.set(xlabel='Tokens before first hallucination onset', ylabel='Test AUROC',
           xticks=DISTANCES, ylim=(.4, 1.))
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/'auroc.png', dpi=200)
    fig.savefig(args.output/'auroc.pdf')
    plt.close(fig)
    lines = ['# First-onset probe results', '',
             'Seed: '+str(args.seed)+'. 95% CIs use matched-pair bootstrap; they do not measure training-seed variance.', '',
             '| Block | Distance | AUROC (95% CI) | AUPRC | Accuracy | F1 | Best epoch |',
             '|---|---|---|---|---|---|---|']
    for r in results:
        lo, hi = r['ci95']['auroc']
        lines.append(f"| {r['layer']} | {r['distance']} | {r['auroc']:.4f} ({lo:.4f}–{hi:.4f}) | {r['auprc']:.4f} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['best_epoch']} |")
    lines += ['', 'This comparison measures future CHAIR object-hallucination association under',
              'oracle first-onset alignment. It does not establish online onset localization,',
              'causality, or within-trajectory monotonic growth of a shared risk score.', '',
              'Negative positions are not aligned to truthful object mentions, so impending',
              'object-mention syntax may contribute to discrimination. Caption lengths are',
              'matched approximately, not exactly. Negative labels mean no CHAIR object',
              'hallucination in the observed output, including length-capped outputs.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'extract', 'train'])
    parser.add_argument('--input', default=str(ROOT/'outputs/llava_coco_train2014/chair_tokens.jsonl'))
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/onset_probe_seed0')
    parser.add_argument('--model-path', default=str(Path.home() / 'models/llava-v1.5-7b'))
    parser.add_argument('--coco-root', default=str(Path.home() / 'dataset/coco'))
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validate-samples', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
