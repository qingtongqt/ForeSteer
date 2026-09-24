"""Common interfaces shared by multimodal model adapters."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenerationOutput:
    """Model-agnostic representation of one autoregressive generation.

    ``token_ids`` are the ids actually returned by generation, rather than a
    second tokenization of ``text``. This matters when hidden states are
    collected for the same decoding trajectory in later stages.
    """

    text: str
    token_ids: List[int]
    tokens: List[str]
    token_texts: List[str]
    token_char_offsets: List[Optional[List[int]]]
    special_tokens_mask: List[int]
    terminated: Optional[bool] = None
    synthetic_prefix_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generation_output(tokenizer, token_ids, stop_ids, stop_text=None):
    """Preserve emitted IDs, with character spans including byte-split tokens."""
    ids = list(token_ids)
    kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = tokenizer.decode(ids, **kwargs)
    terminated = bool(ids and ids[-1] in stop_ids)
    if stop_text and text.endswith(stop_text):
        text = text[:-len(stop_text)]
        terminated = True
    ends = [0]
    for i in range(1, len(ids)+1):
        prefix = tokenizer.decode(ids[:i], **kwargs)
        end = 0
        for a, b in zip(prefix, text):
            if a != b:
                break
            end += 1
        if end < ends[-1]:
            raise ValueError('Tokenizer prefix offsets are not monotonic')
        ends.append(end)
    special_ids = set(tokenizer.all_special_ids) | set(stop_ids)
    special = [int(i in special_ids) for i in ids]
    offsets = [None if s or a == b else [a, b]
               for a, b, s in zip(ends, ends[1:], special)]
    # A Unicode character may span several byte tokens. Its onset belongs to
    # the earliest byte, so give incomplete byte tokens the overlapping span.
    for i, span in enumerate(offsets):
        if span is not None:
            j = i-1
            while j >= 0 and not special[j] and ends[j] == ends[j+1] == span[0]:
                offsets[j] = list(span)
                j -= 1
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens = [t.decode('utf-8', errors='replace') if isinstance(t, bytes) else t for t in tokens]
    return GenerationOutput(text=text, token_ids=ids, tokens=tokens,
                            token_texts=['' if span is None else text[span[0]:span[1]] for span in offsets],
                            token_char_offsets=offsets, special_tokens_mask=special,
                            terminated=terminated, synthetic_prefix_tokens=0)


class LargeMultimodalModel:
    """Minimal interface used by data-generation scripts."""

    model_name = "unknown"

    def __init__(self, device: str = "cuda"):
        self.device = device

    def generate(self, image_path: str, prompt: str) -> GenerationOutput:
        raise NotImplementedError

    def forward(self, image, prompt):
        return ""

def create_hook(feat_list, loc='output'):
    if loc == 'output':
        def hook(module, input, output):
            feat_list.append(output[0])
    else:
        def hook(module, input, output):
            feat_list.append(input[0])
    return hook
