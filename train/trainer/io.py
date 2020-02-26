import sys, os
from .PinkModule.logging import PinkBlackLogger
from omegaconf import OmegaConf


def convert_type(string: str):
    try:
        f = float(string)
        if f.is_integer():
            return int(f)
        else:
            return f
    except ValueError:
        return string


def get_args(default_config: dict):
    import argparse

    parser = argparse.ArgumentParser()

    if not "gpu" in default_config.keys():
        parser.add_argument(f"--gpu", default=None, help="CUDA visible devices : default:None")

    for k, v in default_config.items():
        k = k.lower()
        if isinstance(v, bool):
            if v is True:
                parser.add_argument(f"--{k}", help=f"{k} : default:{v}", action="store_false")
            else:
                parser.add_argument(f"--{k}", help=f"{k} : default:{v}", action="store_true")

        else:
            parser.add_argument(f"--{k}", default=v, help=f"{k} : default:{v}")
    args = parser.parse_args()

    if args.gpu and "gpu" in default_config.keys():
        # Default argument로 gpu를 줬다면 이렇게 세팅
        os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu)})

    for k in default_config.keys():
        k = k.lower()
        val = getattr(args, k)
        if not isinstance(val, bool):
            setattr(args, k, convert_type(str(val)))

    return args


def setup(
    trace=False,
    pdb_on_error=True,
    default_config=None,
    autolog=False,
    autolog_dir="pinkblack_autolog",
):
    """
    :param trace:
    :param pdb_on_error:
    :param default_config: dict or str(yaml file)
    :param autolog:
    :param autolog_dir:
    gpu -> CUDA_VISIBLE_DEVICES
    :return: argparsed config

    Example >>
    ```bash
    CUDA_VISIBLE_DEVICES=1,3 python myscript.py --batch_size 32 --ckpt ckpt.pth --epochs 100 --lr 0.001
    ```
    ```python3
    setup(default_config=dict(gpu="1,3",
                             batch_size=32,
                             lr=1e-3,
                             epochs=100,
                             ckpt="ckpt.pth"))
    ```
    ```bash
    python myscript.py --gpu 1,3 --batch_size 32 --ckpt ckpt.pth --epochs 100 --lr 0.001
    ```

    """
    if trace:
        import backtrace

        backtrace.hook(align=True)

    if pdb_on_error:
        old_hook = sys.excepthook

        def new_hook(type_, value, tb):
            old_hook(type_, value, tb)
            if type_ != KeyboardInterrupt:
                import pdb

                pdb.post_mortem(tb)

        sys.excepthook = new_hook

    args = None
    if default_config is not None:
        if isinstance(default_config, str):
            default_config = OmegaConf.load(default_config)
        args = get_args(default_config)

    import time, datetime

    dt = datetime.datetime.fromtimestamp(time.time())
    dt = datetime.datetime.strftime(dt, f"{os.path.basename(sys.argv[0])}.%Y%m%d_%H%M%S.log")

    if args is not None and hasattr(args, "ckpt"):
        logpath = args.ckpt + "_" + dt
    else:
        logpath = os.path.join(autolog_dir, dt)

    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    if args is not None:
        conf = OmegaConf.create(args.__dict__)
        conf.save(logpath[:-4] + ".yaml")
        conf.save(args.ckpt + ".yaml")

    if autolog:
        fp = open(logpath, "w")
        sys.stdout = PinkBlackLogger(fp, sys.stdout)
        sys.stderr = PinkBlackLogger(fp, sys.stderr)
        print("PinkBlack :: args :", args.__dict__)

    return args


def set_seeds(seed, strict=False):
    """
    strict 가 True이면, Cudnn까지도 deterministic 하게 한다.
    cudnn은 아주 조금 stochastic한 연산 결과를 보여주므로, 정확한 재현이 필요하다면 True로 설정.

    If strict == True, then cudnn backend will be deterministic
    torch.backends.cudnn.deterministic = True
    """
    import random
    import numpy as np
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if strict:
            torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
