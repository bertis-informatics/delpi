def _get_layer_id(name, num_layers):
    if (
        name.startswith("aa_embedding")
        or name.startswith("mod_embedding")
        or name.startswith("meta_embedding")
        or name.startswith("encoder.0")
    ):
        return 0
    elif name.startswith("encoder.2.transformer"):
        return int(name.split(".")[3]) + 1
    else:
        return num_layers


def param_groups_lrd(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):

    param_group_names = {}
    param_groups = {}

    num_layers = len(model.encoder[2].transformer) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = _get_layer_id(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())


def _get_layer_id_for_rt_predictor(name, num_layers):
    if name.startswith("aa_embedding") or name.startswith("mod_embedding"):
        return 0
    elif name.startswith("encoder.1.conv1") or name.startswith("encoder.1.bn1"):
        return 1
    elif name.startswith("encoder.1.layer"):
        return int(name.split(".")[2][-1:]) + 1
    elif name.startswith("encoder.4"):
        return num_layers - 1
    else:
        return num_layers


def param_groups_lrd_for_rt_predictor(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):

    assert (
        model.encoder_type == "cnn_rnn"
    ), "Layer-wise lr decay is only implemented for cnn_rnn encoder."

    param_group_names = {}
    param_groups = {}

    num_layers = len(model.encoder) + 2
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = _get_layer_id_for_rt_predictor(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())
