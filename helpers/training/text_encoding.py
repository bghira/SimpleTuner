from transformers import PretrainedConfig
import os
from accelerate.logging import get_logger
from helpers.models import get_model_config_path
from .state_tracker import StateTracker

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)


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
    try:
        if args.model_family.lower() == "smoldit":
            from transformers import AutoTokenizer

            tokenizer_1 = AutoTokenizer.from_pretrained(
                "EleutherAI/pile-t5-base", pad_token="[PAD]"
            )
            return tokenizer_1, tokenizer_2, tokenizer_3

        tokenizer_kwargs = {
            "pretrained_model_name_or_path": get_model_config_path(
                args.model_family, args.pretrained_model_name_or_path
            ),
            "subfolder": "tokenizer",
            "revision": args.revision,
        }
        is_t5_model = False
        if args.model_family.lower() == "pixart_sigma":
            from transformers import T5Tokenizer

            tokenizer_cls = T5Tokenizer
            is_t5_model = True
        elif args.model_family == "sana":
            from transformers import Gemma2Model, GemmaTokenizerFast

            tokenizer_cls = GemmaTokenizerFast
            is_t5_model = False
            tokenizer_1 = tokenizer_cls.from_pretrained(
                get_model_config_path(
                    args.model_family, args.pretrained_model_name_or_path
                ),
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )
        elif args.model_family.lower() == "kolors":
            from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer

            tokenizer_cls = ChatGLMTokenizer
            tokenizer_1 = tokenizer_cls.from_pretrained(
                get_model_config_path(
                    args.model_family, args.pretrained_model_name_or_path
                ),
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )
        else:
            from transformers import CLIPTokenizer

            tokenizer_1 = CLIPTokenizer.from_pretrained(**tokenizer_kwargs)

        if is_t5_model:
            text_encoder_path = (
                (
                    args.pretrained_t5_model_name_or_path
                    if args.pretrained_t5_model_name_or_path is not None
                    else get_model_config_path(
                        args.model_family, args.pretrained_model_name_or_path
                    )
                ),
            )
            logger.info(
                f"Tokenizer path: {text_encoder_path}, custom T5 model path: {args.pretrained_t5_model_name_or_path} revision: {args.revision}"
            )
            try:
                tokenizer_1 = tokenizer_cls.from_pretrained(
                    text_encoder_path,
                    subfolder="tokenizer",
                    revision=args.revision,
                    use_fast=False,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer 1: {e}, attempting no subfolder"
                )
                tokenizer_1 = T5Tokenizer.from_pretrained(
                    text_encoder_path,
                    subfolder=None,
                    revision=args.revision,
                    use_fast=False,
                )
    except Exception as e:
        import traceback

        logger.warning(
            "Primary tokenizer (CLIP-L/14) failed to load. Continuing to test whether we have just the secondary tokenizer.."
            f"\nError: -> {e}"
            f"\nTraceback: {traceback.format_exc()}"
        )
        if args.model_family in ["sd3"]:
            raise e

    from transformers import T5TokenizerFast

    if args.model_family not in ["pixart_sigma", "kolors", "sana"]:
        try:
            tokenizer_2_cls = CLIPTokenizer
            if args.model_family.lower() == "flux":
                tokenizer_2_cls = T5TokenizerFast
            tokenizer_2 = tokenizer_2_cls.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer_2",
                revision=args.revision,
                use_fast=False,
            )
            if tokenizer_1 is None:
                logger.info("Seems that we are training an SDXL refiner model.")
                StateTracker.is_sdxl_refiner(True)
                if args.validation_using_datasets is None:
                    logger.warning(
                        "Since we are training the SDXL refiner and --validation_using_datasets was not specified, it is now being enabled."
                    )
                    args.validation_using_datasets = True
        except Exception as e:
            logger.warning(
                f"Could not load secondary tokenizer ({'OpenCLIP-G/14' if args.model_family != 'flux' else 'T5 XXL'}). Cannot continue: {e}"
            )
            if args.model_family in ["flux", "sd3"]:
                raise e
        if not tokenizer_1 and not tokenizer_2:
            raise Exception("Failed to load tokenizer")
    else:
        if not tokenizer_1:
            raise Exception("Failed to load tokenizer")

    if args.model_family == "sd3":
        try:
            tokenizer_3 = T5TokenizerFast.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer_3",
                revision=args.revision,
                use_fast=True,
            )
        except:
            raise ValueError(
                "Could not load tertiary tokenizer (T5-XXL v1.1). Cannot continue."
            )
    return tokenizer_1, tokenizer_2, tokenizer_3


def determine_te_path_subfolder(args):
    if args.model_family.lower() == "kolors":
        logger.info("Loading Kolors ChatGLM language model..")
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"
    elif args.model_family.lower() == "smoldit":
        text_encoder_path = "EleutherAI/pile-t5-base"
        text_encoder_subfolder = None
    elif args.model_family.lower() == "flux":
        text_encoder_path = args.pretrained_model_name_or_path
        text_encoder_subfolder = "text_encoder"
    elif args.model_family.lower() == "pixart_sigma":
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
        logger.info("Load CLIP text encoder..")
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
        if args.model_family.lower() == "pixart_sigma":
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
            text_encoder_variant = "fp16"
        elif args.model_family.lower() == "sana":
            logger.info(
                f"Loading Gemma2 language model from {text_encoder_path}/{text_encoder_subfolder}.."
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
