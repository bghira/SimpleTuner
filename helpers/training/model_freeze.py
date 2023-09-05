import logging, os

logger = logging.getLogger('ModelFreeze')
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

def freeze_entire_component(component):
    for name, param in component.named_parameters():
        if hasattr(param, "requires_grad"):
            param.requires_grad = False
    return component


def freeze_text_encoder(args, component):
    if not args.freeze_encoder:
        logging.info(f"Not freezing text encoder. Live dangerously and prosper!")
        return component
    method = args.freeze_encoder_strategy
    first_layer = args.freeze_encoder_before
    last_layer = args.freeze_encoder_after
    total_count = 0
    for name, param in component.named_parameters():
        total_count += 1
        pieces = name.split(".")
        if pieces[1] != "encoder" and pieces[2] != "layers":
            logging.info(f"Ignoring non-encoder layer: {name}")
            continue
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
                logging.debug(
                    f"Froze layer {name} with method {method} and range {first_layer} - {last_layer}"
                )
            else:
                logging.info(
                    f"Ignoring layer that does not mark as gradient capable: {name}"
                )
    logging.info(
        f"Applied {method} method with range {first_layer} - {last_layer} to {total_count} total layers."
    )
    return component
