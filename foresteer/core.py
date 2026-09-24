from contextlib import contextmanager
import math
from pathlib import Path
import torch
from torch import nn


def decoder_layers(adapter):
    """Adapters may override this protocol for any other causal decoder."""
    if hasattr(adapter, 'decoder_layers'):
        return adapter.decoder_layers()
    for path in ('model.layers', 'transformer.h', 'llama_model.model.layers'):
        obj = adapter.model
        try:
            for name in path.split('.'):
                obj = getattr(obj, name)
            return obj
        except AttributeError:
            pass
    raise TypeError('Adapter must implement decoder_layers()')


def discounted_cost(labels, gamma):
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must be in [0, 1]')
    result = torch.zeros(len(labels), dtype=torch.float32)
    future = 0.
    for i in reversed(range(len(labels))):
        future = float(labels[i]) + gamma * future
        result[i] = future
    return result


def windowed_cost(labels, gamma, horizon=16, *, terminated):
    """Return finite-window costs and a mask for fully observed targets.

    Only true terminal states permit zero extension. NaNs deliberately mark
    censored targets so callers cannot accidentally train on partial returns.
    """
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must be in [0, 1]')
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    if not isinstance(terminated, bool):
        raise ValueError('terminated must be an explicit boolean')
    costs = torch.as_tensor(labels, dtype=torch.float32)
    if costs.ndim != 1 or not torch.all((costs == 0) | (costs == 1)):
        raise ValueError('Expected a one-dimensional binary cost sequence')
    target = torch.zeros_like(costs)
    for offset in range(min(horizon, len(costs))):
        target[:len(costs)-offset] += gamma**offset * costs[offset:]
    valid = torch.ones(len(costs), dtype=torch.bool)
    if not terminated:
        valid[max(0, len(costs)-horizon+1):] = False
    target[~valid] = float('nan')
    return target, valid


class RiskPredictor(nn.Module):
    def __init__(self, width, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                 nn.GELU(), nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Steering:
    def __init__(self, adapter, checkpoint, alpha, adaptive=False):
        self.adapter, self.alpha, self.adaptive = adapter, alpha, adaptive
        self.checkpoint = checkpoint
        self.risk_horizon = None
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        if not checkpoint:
            raise ValueError('ForeSteer evaluation requires --risk-checkpoint')
        data = torch.load(checkpoint, map_location='cpu')
        if data.get('schema') not in {'foresteer.risk.v1', 'foresteer.risk.v2'}:
            raise ValueError('Expected a ForeSteer risk-regression checkpoint')
        if data['schema'] == 'foresteer.risk.v2':
            self.risk_horizon = data['risk_horizon']
            if not isinstance(self.risk_horizon, int) or self.risk_horizon < 1:
                raise ValueError('Invalid risk horizon in checkpoint')
        if Path(adapter.args.model_path).expanduser().resolve() != Path(data['model_path']).expanduser().resolve():
            raise ValueError('Risk checkpoint belongs to a different model')
        self.layer = data['layer']
        layers = decoder_layers(adapter)
        if not 1 <= self.layer <= len(layers):
            raise ValueError('Checkpoint layer is outside decoder')
        device = next(layers[self.layer-1].parameters()).device
        self.predictor = RiskPredictor(data['width'], data['hidden']).to(device).eval()
        self.predictor.load_state_dict(data['state_dict'])
        self.predictor.requires_grad_(False)

    def to_metadata(self):
        args = self.adapter.args
        return dict(method='risk_gradient',
                    checkpoint=self.checkpoint, alpha=self.alpha, adaptive=self.adaptive,
                    risk_horizon=self.risk_horizon, 
                    model_path=args.model_path, seed=getattr(args, 'seed', None),
                    max_new_tokens=args.max_length, num_beams=args.num_beams,
                    temperature=args.temperature, batch_size=getattr(args, 'batch_size', 1))

    def hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        # Only the last causal position predicts the next token, including prefill.
        with torch.inference_mode(False), torch.enable_grad():
            x = h[:, -1, :].detach().clone().float().requires_grad_(True)
            risk = self.predictor(x)
            grad, = torch.autograd.grad(risk.sum(), x)
            scale = self.alpha * torch.ones_like(risk)
            delta = scale[:, None] * grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if self.adaptive:
                # Gate negative risk only; zero and positive risk use fixed alpha.
                delta = torch.where((risk.detach() < 0)[:, None], torch.zeros_like(delta), delta)
        changed = h.clone()
        changed[:, -1, :] -= delta.to(h.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    @contextmanager
    def enabled(self):
        handle = None
        try:
            handle = decoder_layers(self.adapter)[self.layer-1].register_forward_hook(self.hook)
            yield
        finally:
            if handle is not None:
                handle.remove()


def add_inference_args(parser):
    parser.add_argument('--risk-checkpoint', required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--adaptive-alpha', action='store_true', help='Skip steering when predicted risk < 0; otherwise use fixed alpha')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)


def validate_inference_args(args):
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    args.model_path = str(Path(args.model_path).expanduser())
    if args.num_beams != 1 or args.temperature != 0:
        raise ValueError('These experiments require greedy decoding (temperature=0, num_beams=1)')
    if not args.risk_checkpoint:
        raise ValueError('ForeSteer evaluation requires --risk-checkpoint')
    args.risk_checkpoint = str(Path(args.risk_checkpoint).expanduser())
    if not Path(args.risk_checkpoint).is_file():
        raise FileNotFoundError(args.risk_checkpoint)
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError('alpha must be finite and positive')


def enable_steering(model, args):
    model.model.eval().requires_grad_(False)
    return Steering(model, args.risk_checkpoint, args.alpha, args.adaptive_alpha)


def chat(model, context, image_path, prompt):
    with context.enabled():
        return model.chat(image_path=image_path, prompt=prompt)


def chat_batch(model, context, image_paths, prompts):
    with context.enabled():
        if hasattr(model, 'chat_batch'):
            return model.chat_batch(image_paths, prompts)
        return [model.chat(image_path=path, prompt=prompt) for path, prompt in zip(image_paths, prompts)]
