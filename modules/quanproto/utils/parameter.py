import quanproto.datasets.config_parser as datasets
import quanproto.models.pipnet.best_params as pipnet
import quanproto.models.protomask.best_params as protomask
import quanproto.models.protopnet.best_params as protopnet
import quanproto.models.protopool.best_params as protopool

protopnet_params = {
    "cub200": protopnet.protopnet_cub200_params | datasets.cub200_params,
    "dogs": protopnet.protopnet_dogs_params | datasets.dogs_params,
    "cars196": protopnet.protopnet_cars196_params | datasets.cars196_params,
    "awa2": protopnet.protopnet_awa2_params | datasets.awa2_params,
    "nico": protopnet.protopnet_nico_params | datasets.nico_params,
}

protopool_params = {
    "cub200": protopool.protopool_cub200_params | datasets.cub200_params,
    "dogs": protopool.protopool_dogs_params | datasets.dogs_params,
    "cars196": protopool.protopool_cars196_params | datasets.cars196_params,
    "awa2": protopool.protopool_awa2_params | datasets.awa2_params,
    "nico": protopool.protopool_nico_params | datasets.nico_params,
}

pipnet_params = {
    "cub200": pipnet.pipnet_cub200_params | datasets.cub200_params,
    "dogs": pipnet.pipnet_dogs_params | datasets.dogs_params,
    "cars196": pipnet.pipnet_cars196_params | datasets.cars196_params,
    "awa2": pipnet.pipnet_awa2_params | datasets.awa2_params,
    "nico": pipnet.pipnet_nico_params | datasets.nico_params,
}

protomask_params = {
    "cub200_sam2": protomask.protomask_cub200_sam2_params | datasets.cub200_params,
    "cub200_slit": protomask.protomask_cub200_slit_params | datasets.cub200_params,
    "dogs_sam2": protomask.protomask_dogs_sam2_params | datasets.dogs_params,
    "dogs_slit": protomask.protomask_dogs_slit_params | datasets.dogs_params,
    "cars196_sam2": protomask.protomask_cars196_sam2_params | datasets.cars196_params,
    "cars196_slit": protomask.protomask_cars196_slit_params | datasets.cars196_params,
}
