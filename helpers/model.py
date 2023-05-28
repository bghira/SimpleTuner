import logging

def freeze_encoder(text_encoder):
    first_unfrozen_layer = 17
    final_unfrozen_layer = 22
    total_count = 0
    for name, param in text_encoder.named_parameters():
        total_count += 1
        pieces = name.split(".")
        if pieces[1] != "encoder" and pieces[2] != "layers":
            logging.info(f"Ignoring non-encoder layer: {name}")
            continue
        logging.info(f'Pieces: {pieces}')
        current_layer = int(pieces[3])
        if (
            current_layer <= first_unfrozen_layer or current_layer >= final_unfrozen_layer
        ):  # we freeze the early and late layers.
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False
                logging.info(f'Froze layer because {current_layer} <= {first_unfrozen_layer} and {current_layer} >= {final_unfrozen_layer}: {name}')
            else:
                logging.info(f'Ignoring layer that does not mark as gradient capable: {name}')
    logging.info(
        f"Thawed text encoder layers between {first_unfrozen_layer} to {final_unfrozen_layer} (exclusive) out of {total_count} total discovered."
    )
    return text_encoder