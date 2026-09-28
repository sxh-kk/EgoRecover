from yacs.config import CfgNode


def cfg_to_dict(cfg):
    cfg_dict = {}
    for k, v in cfg.items():
        if isinstance(v, CfgNode):
            cfg_dict[k] = cfg_to_dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict
