from quanproto.models.config_parser import model_loading_fn_dict
from quanproto.models.pipnet import pipnet
from quanproto.models.protomask import protomask
from quanproto.models.protopnet import protopnet
from quanproto.models.protopool import protopool

original_model_dict = {
    "protopnet": protopnet.ProtoPNet,
    "protopool": protopool.ProtoPool,
    "pipnet": pipnet.PIPNet,
    "protomask": protomask.ProtoMask,
}

from quanproto.explanations.prp.models import pipnet, protopnet, protopool, protomask

prp_explanation_fn_dict = {
    "protopnet": protopnet.ProtoPNetPRP,
    "protopool": protopool.ProtoPoolPRP,
    "pipnet": pipnet.PIPNetPRP,
    "protomask": protomask.ProtoMaskPRP,
}

from quanproto.explanations.upscale.models import pipnet, protopnet, protopool

upscale_explanation_fn_dict = {
    "protopnet": protopnet.ProtoPNetUpscale,
    "protopool": protopool.ProtoPoolUpscale,
    "pipnet": pipnet.PIPNetUpscale,
}

from quanproto.explanations.mask.models import protomask

mask_explanation_fn_dict = {
    "protomask": protomask.ProtoMaskMask,
}


def load_model(config, explanation_name, state_dict):

    model_name = config["model"]

    if explanation_name == None:
        if model_name not in original_model_dict:
            raise ValueError("Model not implemented for explanation")

        model = model_loading_fn_dict[model_name](
            config, state_dict, original_model_dict[model_name]
        )
        return model

    if explanation_name == "prp":
        if model_name not in prp_explanation_fn_dict:
            raise ValueError("Model not implemented for explanation")

        model = model_loading_fn_dict[model_name](
            config, state_dict, prp_explanation_fn_dict[model_name]
        )
        model.canonize((3, 224, 224), config["features"])
        return model

    if explanation_name == "upscale":
        if model_name not in upscale_explanation_fn_dict:
            raise ValueError("Model not implemented for explanation")

        model = model_loading_fn_dict[model_name](
            config, state_dict, upscale_explanation_fn_dict[model_name]
        )
        return model
    if explanation_name == "mask":
        if model_name not in mask_explanation_fn_dict:
            raise ValueError("Model not implemented for explanation")

        model = model_loading_fn_dict[model_name](
            config, state_dict, mask_explanation_fn_dict[model_name]
        )
        return model

    raise ValueError("Explanation name not recognized")
