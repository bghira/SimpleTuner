import logging, os

logger = logging.getLogger("ModelFreeze")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def apply_bitfit_freezing(model):
    logger.debug("Applying BitFit freezing strategy for u-net tuning.")
    for name, param in model.named_parameters():
        # Freeze everything that's not a bias
        if "bias" not in name:
            param.requires_grad = False
        else:
            # Unfreeze biases
            param.requires_grad = True
    return model


def freeze_entire_component(component):
    for name, param in component.named_parameters():
        if hasattr(param, "requires_grad"):
            param.requires_grad = False
    return component


def freeze_text_encoder(args, component):
    from transformers import T5EncoderModel

    if not args.freeze_encoder or type(component) is T5EncoderModel:
        logger.info(f"Not freezing text encoder. Live dangerously and prosper!")
        return component
    method = args.freeze_encoder_strategy
    first_layer = args.freeze_encoder_before
    last_layer = args.freeze_encoder_after
    total_count = 0
    for name, param in component.named_parameters():
        total_count += 1
        pieces = name.split(".")
        if pieces[1] != "encoder" and pieces[2] != "layers":
            logger.info(f"Ignoring non-encoder layer: {name}")
            continue
        else:
            logger.debug(f"Freezing layer: {name}, which has keys: {pieces}")
        current_layer = int(pieces[3])

        freeze_param = False
        if method == "between":
            freeze_param = current_layer > first_layer or current_layer < last_layer
        elif method == "outside":
            freeze_param = first_layer <= current_layer <= last_layer
        elif method == "before":
            freeze_param = current_layer < first_layer
        elif method == "after":
            freeze_param = current_layer > last_layer
        else:
            raise ValueError(
                f"Invalid method {method}. Choose between 'between', 'outside', 'before' or 'after'."
            )

        if freeze_param:
            if hasattr(param, "requires_grad"):
                param.requires_grad = False
                # logger.debug(
                #     f"Froze layer {name} with method {method} and range {first_layer} - {last_layer}"
                # )
            else:
                # logger.info(
                #     f"Ignoring layer that does not mark as gradient capable: {name}"
                # )
                pass
    logger.info(
        f"Applied {method} method with range {first_layer} - {last_layer} to {total_count} total layers."
    )
    return component
