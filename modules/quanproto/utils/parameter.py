import quanproto.datasets.config_parser as datasets
import quanproto.models.pipnet.best_params as pipnet
import quanproto.models.protopnet.best_params as protopnet
import quanproto.models.protopool.best_params as protopool

protopnet_params = {
    "cub200": protopnet.protopnet_cub200_params | datasets.cub200_params,
    "cars196": protopnet.protopnet_cars196_params | datasets.cars196_params,
    "awa2": protopnet.protopnet_awa2_params | datasets.awa2_params,
    "nico": protopnet.protopnet_nico_params | datasets.nico_params,
}

protopool_params = {
    "cub200": protopool.protopool_cub200_params | datasets.cub200_params,
    "cars196": protopool.protopool_cars196_params | datasets.cars196_params,
    "awa2": protopool.protopool_awa2_params | datasets.awa2_params,
    "nico": protopool.protopool_nico_params | datasets.nico_params,
}

pipnet_params = {
    "cub200": pipnet.pipnet_cub200_params | datasets.cub200_params,
    "cars196": pipnet.pipnet_cars196_params | datasets.cars196_params,
    "awa2": pipnet.pipnet_awa2_params | datasets.awa2_params,
    "nico": pipnet.pipnet_nico_params | datasets.nico_params,
}
