def determine_adapter_target_modules(args, unet, transformer):
    if unet is not None:
        return ["to_k", "to_q", "to_v", "to_out.0"]
    elif transformer is not None:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        if args.flux and args.flux_lora_target == "all":
            # target_modules = mmdit layers here
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out.0",
            ]
        elif args.flux_lora_target == "context":
            # i think these are the text input layers.
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out.0",
            ]
        elif args.flux_lora_target == "all+ffs":
            target_modules = [
                "to_k",
                "to_q",
                "to_v",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out.0",
                "ff.0",
                "ff.2",
                "ff_context.0",
                "ff_context.2",
                "proj_mlp",
                "proj_out",
            ]
        elif args.flux_lora_target == "ai-toolkit":
            # from ostris' ai-toolkit, possibly required to continue finetuning one.
            target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
                "norm.linear",
                "norm1.linear",
                "norm1_context.linear",
                "proj_mlp",
                "proj_out",
            ]

        return target_modules
