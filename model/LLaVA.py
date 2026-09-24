from PIL import Image

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from model.base import GenerationOutput, LargeMultimodalModel


def add_diffusion_noise(image_tensor, noise_step): # from VCD
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step)

    return image_tensor_cd


def create_hook(feat_list, loc='output'):
    if loc == 'output':
        def hook(module, input, output):
            feat_list.append(output[0])
    else:
        def hook(module, input, output):
            feat_list.append(input[0])
    return hook


class LLaVA(LargeMultimodalModel):
    model_name = "llava"

    def __init__(self, args):
        super(LLaVA, self).__init__(device=getattr(args, "device", "cuda"))
        load_8bit = False
        load_4bit = False

        # Load Model
        disable_torch_init()

        model_name = get_model_name_from_path(args.model_path)
        if "finetune-lora" in args.model_path:
            model_base = "liuhaotian/llava-v1.5-7b"
        elif "lora" in args.model_path:
            model_base = "lmsys/vicuna-7b-v1.5"
        else:
            model_base = None
        self.args = args
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=self.device,
        )
        self.conv_mode = "llava_v1"
        self.num_lm_layers = self.model.config.num_hidden_layers
        self.num_lm_hidden_size = self.model.config.hidden_size
        self.lm_head = self.model.lm_head

    def refresh_chat(self):
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles

    def _image_tensor(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return process_images([image], self.image_processor, self.model.config).to(
            self.device,
            dtype=torch.float16,
        )

    @torch.no_grad()
    def _basic_forward(self, noise_flag, image_path, prompt, answer=None, return_dict=False):
        self.refresh_chat()
        image_tensor = self._image_tensor(image_path)
        if noise_flag:
            image_tensor = add_diffusion_noise(image_tensor, 500)

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], answer)

        conv_prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).to(self.device)

        outputs = self.model(
            input_ids,
            images=image_tensor,
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict)

        return outputs

    @torch.no_grad()
    def _generate_token_ids(self, image_path, prompt):
        self.refresh_chat()
        image_tensor = self._image_tensor(image_path)

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        conv_prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        generate_kwargs = {
            "images": image_tensor,
            "do_sample": self.args.temperature > 0,
            "num_beams": self.args.num_beams,
            "max_new_tokens": self.args.max_length,
            "use_cache": True,
            "stopping_criteria": [stopping_criteria],
            "return_dict_in_generate": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "output_scores": False,
        }
        if self.args.temperature > 0:
            generate_kwargs["temperature"] = self.args.temperature
            if self.args.top_p is not None:
                generate_kwargs["top_p"] = self.args.top_p
            if self.args.top_k is not None:
                generate_kwargs["top_k"] = self.args.top_k

        outputs = self.model.generate(
            input_ids,
            **generate_kwargs,
        )

        sequence_ids = outputs.sequences[0]
        # Most LLaVA versions generate from inputs_embeds and return only new
        # ids. Keep compatibility with versions returning prompt + response.
        if (sequence_ids.numel() >= input_ids.shape[1]
                and torch.equal(sequence_ids[:input_ids.shape[1]], input_ids[0])):
            sequence_ids = sequence_ids[input_ids.shape[1]:]
        return sequence_ids.tolist(), stop_str

    def _decode_with_offsets(self, token_ids):
        decode_kwargs = {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        prefixes = [""]
        for end in range(1, len(token_ids) + 1):
            prefixes.append(self.tokenizer.decode(token_ids[:end], **decode_kwargs))

        text = prefixes[-1]
        offsets = []
        token_texts = []
        for previous, current in zip(prefixes, prefixes[1:]):
            if not current.startswith(previous):
                raise RuntimeError(
                    "Tokenizer prefix decoding changed previously decoded text; "
                    "cannot construct reliable token character offsets."
                )
            piece_text = current[len(previous):]
            token_texts.append(piece_text)
            offsets.append([len(previous), len(current)] if piece_text else None)
        return text, token_texts, offsets

    @torch.no_grad()
    def generate(self, image_path, prompt):
        """Greedily generate and retain the exact autoregressive token ids."""
        token_ids, stop_str = self._generate_token_ids(image_path, prompt)
        text, token_texts, offsets = self._decode_with_offsets(token_ids)

        # llava_v1 normally uses the special </s> token, which is already
        # removed by skip_special_tokens. Handle textual separators too.
        if stop_str and text.endswith(stop_str):
            text = text[:-len(stop_str)]
            text_end = len(text)
            offsets = [
                None if span is None or span[0] >= text_end
                else [span[0], min(span[1], text_end)]
                for span in offsets
            ]
            token_texts = [
                "" if span is None else text[span[0]:span[1]]
                for span in offsets
            ]

        special_ids = set(self.tokenizer.all_special_ids)
        return GenerationOutput(
            text=text,
            token_ids=token_ids,
            tokens=self.tokenizer.convert_ids_to_tokens(token_ids),
            token_texts=token_texts,
            token_char_offsets=offsets,
            special_tokens_mask=[int(token_id in special_ids) for token_id in token_ids],
        )

    @torch.no_grad()
    def chat_batch(self, image_paths, prompts):
        """Left-padded greedy evaluation; each batch row is an independent image."""
        if len(image_paths) != len(prompts) or not prompts:
            raise ValueError("Expected equal nonempty image/prompt batches")
        sequences = []
        for prompt in prompts:
            self.refresh_chat()
            token = DEFAULT_IMAGE_TOKEN
            if self.model.config.mm_use_im_start_end:
                token = DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN
            self.conv.append_message(self.conv.roles[0], token + "\n" + prompt)
            self.conv.append_message(self.conv.roles[1], None)
            sequences.append(tokenizer_image_token(self.conv.get_prompt(), self.tokenizer,
                             IMAGE_TOKEN_INDEX, return_tensors='pt'))
        width = max(len(ids) for ids in sequences)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, -len(sequence):] = sequence.to(self.device)
            mask[row, -len(sequence):] = 1
        images = torch.cat([self._image_tensor(path) for path in image_paths])
        previous_side = getattr(self.model.config, 'tokenizer_padding_side', None)
        self.model.config.tokenizer_padding_side = 'left'
        try:
            output = self.model.generate(ids, attention_mask=mask, images=images,
                do_sample=False, num_beams=1, max_new_tokens=self.args.max_length,
                use_cache=True, pad_token_id=pad, return_dict_in_generate=True)
        finally:
            if previous_side is None:
                delattr(self.model.config, 'tokenizer_padding_side')
            else:
                self.model.config.tokenizer_padding_side = previous_side
        responses = []
        for row, sequence in enumerate(output.sequences):
            if len(sequence) >= width and torch.equal(sequence[:width], ids[row]):
                sequence = sequence[width:]
            responses.append(self.tokenizer.decode(sequence, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False).strip())
        return responses

    @torch.no_grad()
    def chat(self, image_path, prompt, answer=None, return_dict=False):
        """Return the generated response text."""
        if answer is not None or return_dict:
            raise ValueError(
                "LLaVA.chat requires answer=None and return_dict=False; "
                "use _basic_forward for teacher forcing"
            )
        return self.generate(image_path, prompt).text

    def register_hooks(self):
        self.model.attn_heads, self.model.attn_residual, self.model.mlp_residual, self.model.vit_satt = [], [], [], []
        attn_head_hook = create_hook(self.model.attn_heads, loc='input')
        attn_residual_hook = create_hook(self.model.attn_residual)
        mlp_residual_hook = create_hook(self.model.mlp_residual)
        self.hooks = []
        for layer in self.model.base_model.layers:
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(attn_head_hook))
            self.hooks.append(layer.self_attn.register_forward_hook(attn_residual_hook))
            self.hooks.append(layer.mlp.register_forward_hook(mlp_residual_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_activations(self, image_path, prompt, answer=None, noise_flag=False):
        self.register_hooks()
        outputs = self._basic_forward(noise_flag, image_path, prompt, answer, return_dict=True)
        attn_heads = torch.cat(self.model.attn_heads).reshape(32, -1, 32, 128)   # [32, seq_len, 4096] -> [32, seq_len, 32, 128]
        attn_residual = torch.cat(self.model.attn_residual)   # [32, seq_len, 4096]
        mlp_residual = torch.stack(self.model.mlp_residual)   # [32, seq_len, 4096]
        hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
        self.remove_hooks()
        return hidden_states, mlp_residual, attn_residual, attn_heads
