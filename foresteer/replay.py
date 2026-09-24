"""Exact-token replay. New adapters implement risk_inputs(image_path, prompt)."""
import torch
import inspect


def risk_inputs(adapter, image_path, prompt):
    if hasattr(adapter, 'risk_inputs'):
        return adapter.risk_inputs(image_path, prompt)
    name = adapter.args.model_name.lower().replace('_', '-')
    if name.startswith('llava'):
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token
        adapter.refresh_chat()
        token = DEFAULT_IMAGE_TOKEN
        if adapter.model.config.mm_use_im_start_end:
            token = DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN
        adapter.conv.append_message(adapter.conv.roles[0], token+'\n'+prompt)
        adapter.conv.append_message(adapter.conv.roles[1], None)
        ids = tokenizer_image_token(adapter.conv.get_prompt(), adapter.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors='pt')[None].to(adapter.device)
        image = adapter._image_tensor(image_path)
        _, _, _, _, embeds, _ = adapter.model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, image)
        return adapter.model, embeds, adapter.model.get_model().embed_tokens
    if name == 'minigpt4':
        from PIL import Image
        adapter.refresh_chat()
        images = []
        adapter.Chat.upload_img(Image.open(image_path).convert('RGB'), adapter.Chat_state, images)
        adapter.Chat.encode_img(images)
        adapter.Chat.ask(prompt, adapter.Chat_state)
        adapter.Chat_state.append_message(adapter.Chat_state.roles[1], None)
        embeds = adapter.model.get_context_emb(adapter.Chat_state.get_prompt(), images)
        return adapter.model.llama_model, embeds, adapter.model.llama_model.get_input_embeddings()
    raise NotImplementedError('Adapter must implement risk_inputs(image_path, prompt)')


@torch.no_grad()
def replay(adapter, record, image_path, layers, validate=False):
    from .core import decoder_layers
    blocks = decoder_layers(adapter)
    if not layers or len(set(layers)) != len(layers) or any(l < 1 or l > len(blocks) for l in layers):
        raise ValueError('Layers must be unique 1-based decoder block indices')
    raw = record['tokenization']['token_ids']
    labels = record['hallucination_start_labels']
    if not raw or any(v not in (0, 1) for v in labels):
        raise ValueError('Expected nonempty token trajectory and binary onset labels')
    if len(raw) != len(labels):
        raise ValueError('Token/label length mismatch')
    # Stored LLaVA generation has a synthetic BOS, not an emitted action.
    skip = int(adapter.args.model_name.lower().startswith('llava') and raw[0] == adapter.tokenizer.bos_token_id)
    skip = record['tokenization'].get('synthetic_prefix_tokens', skip)
    if not isinstance(skip, int) or not 0 <= skip < len(raw) or any(labels[:skip]):
        raise ValueError('Invalid synthetic token prefix')
    ids, labels = raw[skip:], labels[skip:]
    model, prompt, embedding = risk_inputs(adapter, image_path, record['prompt'])
    capture, handles = {}, []
    for layer in layers:
        def hook(module, inputs, output, layer=layer):
            h = output[0] if isinstance(output, tuple) else output
            capture[layer] = h[0].detach().cpu().half()
        handles.append(decoder_layers(adapter)[layer-1].register_forward_hook(hook))
    try:
        response = torch.tensor([ids[:-1]], device=prompt.device, dtype=torch.long)
        extra = embedding if isinstance(embedding, dict) else {}
        if embedding is not None and not isinstance(embedding, dict):
            full = torch.cat([prompt, embedding(response)], dim=1)
            model(inputs_embeds=full, attention_mask=torch.ones(full.shape[:2], device=full.device, dtype=torch.long),
                  use_cache=False, return_dict=True)
        else:
            full = torch.cat([prompt, response], dim=1)
            model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False, return_dict=True, **extra)
        start = capture[layers[0]].shape[0]-len(ids)
        features = {l: capture[l][start:start+len(ids)].clone() for l in layers}
        if any(len(x) != len(labels) for x in features.values()):
            raise RuntimeError('Causal replay alignment failed')
        if validate:
            prompt_mask = torch.ones(prompt.shape[:2], device=prompt.device, dtype=torch.long)
            if embedding is not None and not isinstance(embedding, dict):
                out = model(inputs_embeds=prompt, attention_mask=prompt_mask, use_cache=True, return_dict=True)
            else:
                out = model(input_ids=prompt, attention_mask=prompt_mask, use_cache=True, return_dict=True, **extra)
            # Count expanded visual positions, without assuming a model's KV
            # tensor layout (Qwen and LLaMA use different cache axes).
            prefill_length = capture[layers[0]].shape[0]
            accepts_position_ids = 'position_ids' in inspect.signature(model.forward).parameters
            if int(out.logits[0,-1].argmax()) != ids[0]:
                raise RuntimeError('Replay does not reproduce first greedy token')
            for i in range(min(len(ids), 8)):
                for layer in layers:
                    a, b = features[layer][i].float(), capture[layer][-1].float()
                    if torch.nn.functional.cosine_similarity(a[None],b[None]).item() < .999 or (a-b).norm()/a.norm().clamp_min(1e-8) > .05:
                        raise RuntimeError(f'KV replay mismatch at layer {layer}, action {i}')
                if i+1 < min(len(ids), 8):
                    position_kwargs = {}
                    if accepts_position_ids:
                        position_kwargs['position_ids'] = torch.tensor([[prefill_length+i]], device=prompt.device)
                    out = model(input_ids=torch.tensor([[ids[i]]],device=prompt.device),
                                attention_mask=torch.ones((1, prefill_length+i+1), device=prompt.device, dtype=torch.long),
                                past_key_values=out.past_key_values, use_cache=True, return_dict=True,
                                **position_kwargs, **extra)
        return features, labels
    finally:
        for handle in handles:
            handle.remove()
