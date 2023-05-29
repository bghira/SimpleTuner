import logging

def freeze_encoder(text_encoder, method='between', first_layer=17, last_layer=22):
    total_count = 0
    for name, param in text_encoder.named_parameters():
        total_count += 1
        pieces = name.split(".")
        if pieces[1] != "encoder" and pieces[2] != "layers":
            logging.info(f"Ignoring non-encoder layer: {name}")
            continue
        logging.info(f'Pieces: {pieces}')
        current_layer = int(pieces[3])

        freeze_param = False
        if method == 'between':
            freeze_param = current_layer > first_layer or current_layer < last_layer
        elif method == 'outside':
            freeze_param = first_layer <= current_layer <= last_layer
        elif method == 'before':
            freeze_param = current_layer < first_layer
        elif method == 'after':
            freeze_param = current_layer > first_layer
        else:
            raise ValueError(f"Invalid method {method}. Choose between 'between', 'outside', 'before' or 'after'.")

        if freeze_param:
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False
                logging.info(f'Froze layer {name} with method {method} and range {first_layer} - {last_layer}')
            else:
                logging.info(f'Ignoring layer that does not mark as gradient capable: {name}')
    logging.info(
        f"Applied {method} method with range {first_layer} - {last_layer} to {total_count} total layers."
    )
    return text_encoder