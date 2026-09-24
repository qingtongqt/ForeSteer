def build_model(args):
    """Build a model adapter behind the common generation interface."""
    normalized_name = args.model_name.lower().replace("_", "-")
    if normalized_name in {"llava", "llava-7b", "llava-v1.5-7b"}:
        from .LLaVA import LLaVA
        model = LLaVA(args)
    elif normalized_name == "minigpt4":
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif normalized_name in {"mplug-owl2", "mplugowl2"}:
        from .mPLUG_Owl2 import mPLUG_Owl2
        model = mPLUG_Owl2(args)
    elif normalized_name in {"qwen-vl-chat", "qwenvlchat", "qwenvl", "qwen-vl"}:
        from .Qwen_VL_Chat import Qwen_VL_Chat
        model = Qwen_VL_Chat(args)
    else:
        raise ValueError(f"Unsupported model adapter: {args.model_name}")
        
    return model
