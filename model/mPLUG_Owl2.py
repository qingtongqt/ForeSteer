import torch
from PIL import Image

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from model.base import LargeMultimodalModel, create_hook, generation_output


class mPLUG_Owl2(LargeMultimodalModel):
    def __init__(self, args):
        super(mPLUG_Owl2, self).__init__(device=getattr(args, "device", "cuda"))
        model_path = args.model_path   # 'MAGAer13/mplug-owl2-llama2-7b'
        model_name = get_model_name_from_path(model_path).replace('mplug-owl2', 'mplug_owl2')

        self.args = args
        # Each shard owns one visible GPU; do not auto-partition visual modules.
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, load_8bit=False, load_4bit=False,
            device=self.device, device_map={"": self.device})

    def decoder_layers(self):
        return self.model.get_model().layers

    def risk_inputs(self, image_path, prompt):
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        image = image.resize((max(image.size), max(image.size)))
        tensor = process_images([image], self.image_processor).to(self.model.device, dtype=torch.float16)
        conv = conv_templates['mplug_owl2'].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors='pt')[None].to(self.model.device)
        return self.model, ids, {'images': tensor}

    @torch.no_grad()
    def generate(self, image_path, prompt):
        _, input_ids, extra = self.risk_inputs(image_path, prompt)
        stop_str = conv_templates['mplug_owl2'].sep2
        stopping = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        kwargs = dict(use_cache=True, max_new_tokens=self.args.max_length,
                      stopping_criteria=[stopping], do_sample=self.args.temperature > 0,
                      num_beams=self.args.num_beams, return_dict_in_generate=True)
        if self.args.temperature > 0:
            kwargs['temperature'] = self.args.temperature
            if self.args.top_p is not None:
                kwargs['top_p'] = self.args.top_p
            if self.args.top_k is not None:
                kwargs['top_k'] = self.args.top_k
        output = self.model.generate(input_ids, **extra, **kwargs)
        sequence = output.sequences[0]
        if not torch.equal(sequence[:input_ids.shape[1]], input_ids[0]):
            raise RuntimeError('mPLUG generation did not preserve input token prefix')
        ids = sequence[input_ids.shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        return generation_output(self.tokenizer, ids, stop_ids, stop_str)

    @torch.no_grad()
    def chat(self, image_path, prompt):
        return self.generate(image_path, prompt).text.strip()

    @torch.no_grad()
    def _basic_forward(self, image_path, prompt, answer=None, return_dict=False):
        image = Image.open(image_path).convert("RGB")
        max_edge = max(image.size)  # Match the square image preprocessing used by the model.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv = conv_templates["mplug_owl2"].copy()

        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], answer)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        outputs = self.model(
            input_ids,
            images=image_tensor,
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict)
        return outputs

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

    def get_activations(self, image_path, prompt, answer=None):
        self.register_hooks()
        outputs = self._basic_forward(image_path, prompt, answer, return_dict=True)
        attn_heads = torch.cat(self.model.attn_heads).reshape(32, -1, 32, 128)   # [32, seq_len, 4096] -> [32, seq_len, 32, 128]
        attn_residual = torch.cat(self.model.attn_residual)   # [32, seq_len, 4096]
        mlp_residual = torch.stack(self.model.mlp_residual)   # [32, seq_len, 4096]
        hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
        self.remove_hooks()
        return hidden_states, mlp_residual, attn_residual, attn_heads
