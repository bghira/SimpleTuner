from transformers import PretrainedConfig
import os
from accelerate.logging import get_logger
from helpers.models.common import get_model_config_path
from .state_tracker import StateTracker

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)

t5_only_models = ["pixart_sigma", "ltxvideo", "wan"]
# also with three text encoders
models_with_two_text_encoders = ["sdxl", "sd3", "flux"]
models_with_three_text_encoders = ["sd3"]


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str,
    args,
    subfolder: str = "text_encoder",
):
    if args.model_family.lower() == "smoldit":
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    elif model_class == "UMT5EncoderModel":
        from transformers import UMT5EncoderModel

        return UMT5EncoderModel
    elif model_class == "ChatGLMModel":
        from diffusers.pipelines.kolors.text_encoder import ChatGLMModel

        return ChatGLMModel
    elif model_class == "Gemma2Model":
        from transformers import Gemma2Model

        return Gemma2Model
    else:
        raise ValueError(f"{model_class} is not supported.")


def get_tokenizers(args):
    tokenizer_1, tokenizer_2, tokenizer_3 = None, None, None
    if args.model_family.lower() == "smoldit":
        from transformers import AutoTokenizer

        tokenizer_1 = AutoTokenizer.from_pretrained(
            "EleutherAI/pile-t5-base", pad_token="[PAD]"
        )
        return tokenizer_1, tokenizer_2, tokenizer_3

    return tokenizer_1, tokenizer_2, tokenizer_3


def determine_te_path_subfolder(args):
    if args.model_family.lower() == "kolors":
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"
    elif args.model_family.lower() == "smoldit":
        text_encoder_path = "EleutherAI/pile-t5-base"
        text_encoder_subfolder = None
    elif args.model_family.lower() == "flux":
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"
    elif args.model_family.lower() in ["ltxvideo", "pixart_sigma"]:
        text_encoder_path = (
            args.pretrained_t5_model_name_or_path
            if args.pretrained_t5_model_name_or_path is not None
            else args.pretrained_model_name_or_path
        )
        # Google's version of the T5 XXL model doesn't have a subfolder :()
        text_encoder_subfolder = "text_encoder"
    else:
        # sdxl and sd3 use the sd 1.5 clip-L/14 as number one.
        # sd2.x uses openclip vit-H/14
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"

    return text_encoder_path, text_encoder_subfolder


def load_tes(
    args,
    text_encoder_classes,
    tokenizers,
    weight_dtype,
    text_encoder_path,
    text_encoder_subfolder,
):
    text_encoder_cls_1, text_encoder_cls_2, text_encoder_cls_3 = text_encoder_classes
    tokenizer_1, tokenizer_2, tokenizer_3 = tokenizers
    text_encoder_1, text_encoder_2, text_encoder_3 = None, None, None
    text_encoder_variant = args.variant

    if tokenizer_1 is not None and not args.model_family == "smoldit":
        if args.model_family.lower() in t5_only_models:
            logger.info(
                f"Loading T5-XXL v1.1 text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        elif args.model_family.lower() == "flux":
            logger.info(
                f"Loading OpenAI CLIP-L text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        elif args.model_family.lower() == "kolors":
            logger.info(
                f"Loading ChatGLM language model from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        else:
            logger.info(
                f"Loading CLIP text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        text_encoder_1 = text_encoder_cls_1.from_pretrained(
            text_encoder_path,
            subfolder=text_encoder_subfolder,
            revision=args.revision,
            variant=text_encoder_variant,
            torch_dtype=weight_dtype,
        )
    elif args.model_family.lower() == "smoldit":
        text_encoder_1 = text_encoder_cls_1.from_pretrained(
            "EleutherAI/pile-t5-base",
            torch_dtype=weight_dtype,
        ).encoder

    if tokenizer_2 is not None:
        if args.model_family.lower() == "flux":
            logger.info(
                f"Loading T5 XXL v1.1 text encoder from {args.pretrained_model_name_or_path}/text_encoder_2.."
            )
        else:
            logger.info("Loading LAION OpenCLIP-G/14 text encoder..")
        text_encoder_2 = text_encoder_cls_2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
            torch_dtype=weight_dtype,
            variant=args.variant,
        )
    if tokenizer_3 is not None and args.model_family == "sd3":
        logger.info("Loading T5-XXL v1.1 text encoder..")
        text_encoder_3 = text_encoder_cls_3.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            torch_dtype=weight_dtype,
            revision=args.revision,
            variant=args.variant,
        )

    for te in [text_encoder_1, text_encoder_2, text_encoder_3]:
        if te is None:
            continue
        te.eval()

    return text_encoder_variant, text_encoder_1, text_encoder_2, text_encoder_3
