import torch


def gan_d_loss(
    discriminator,
    teacher_transformer,
    sample_fake,
    sample_real,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask,
    weight,
    discriminator_head_stride,
):
    loss = 0.0
    # collate sample_fake and sample_real
    with torch.no_grad():
        (_, fake_features, fake_features_ori) = teacher_transformer(
            sample_fake,
            timestep,
            encoder_hidden_states,
            output_features=True,
            output_features_stride=2,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )
        (_, real_features, real_features_ori) = teacher_transformer(
            sample_real,
            timestep,
            encoder_hidden_states,
            output_features=True,
            output_features_stride=2,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )

    fake_outputs = discriminator(fake_features_ori)
    real_outputs = discriminator(real_features_ori)
    for fake_output, real_output in zip(fake_outputs, real_outputs):
        loss += (
            torch.mean(weight * torch.relu(fake_output.float() + 1))
            + torch.mean(weight * torch.relu(1 - real_output.float()))
        ) / (discriminator.head_num * discriminator.num_h_per_head)
    return loss


def gan_g_loss(
    discriminator,
    teacher_transformer,
    sample_fake,
    sample_real,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask,
    weight,
    discriminator_head_stride,
):
    loss = 0.0
    (_, features, features_ori) = teacher_transformer(
        sample_fake,
        timestep,
        encoder_hidden_states,
        output_features=True,
        output_features_stride=2,
        return_dict=False,
        final_layer=True,
        unpachify_layer=True,
        student=False,
    )

    with torch.no_grad():
        (_, features_real, features_real_ori) = teacher_transformer(
            sample_real,
            timestep,
            encoder_hidden_states,
            output_features=True,
            output_features_stride=2,
            return_dict=False,
            final_layer=True,
            unpachify_layer=True,
            student=False,
        )

    loss_feat = torch.nn.functional.mse_loss(features, features_real) * 10.0

    fake_outputs = discriminator(
        features_ori,
    )
    for fake_output in fake_outputs:
        loss += torch.mean(weight * torch.relu(1 - fake_output.float())) / (
            discriminator.head_num * discriminator.num_h_per_head
        )
    loss = loss * 5.0
    return loss + loss_feat
