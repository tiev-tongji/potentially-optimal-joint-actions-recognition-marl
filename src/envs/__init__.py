from .stag_hunt import StagHunt
from .matrix_game import OneStepMatrixGame
from .multiagentenv import MultiAgentEnv
from functools import partial
import sys
import os
from colorama import Fore, Style, init
from termcolor import colored
from .highway_env.IntersecEnv import IntersecEnv, IntersecEnvRay

# Initialize colorama for cross-platform support
init(autoreset=True)


try:
    smac = True
    from .smac_v1 import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False

try:
    smacv2 = True
    from .smac_v2 import StarCraft2Env2Wrapper
except Exception as e:
    print(e)
    smacv2 = False


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
# Register environments
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["intersec"] = partial(env_fn, env=IntersecEnv)
REGISTRY["intersec_ray"] = partial(env_fn, env=IntersecEnvRay)

home_dir = os.path.expanduser("~")

if smac:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH", os.path.join(home_dir, "StarCraftII"))
else:
    print(Fore.RED + "SMAC V1 is not supported...")

if smacv2:
    REGISTRY["sc2_v2"] = partial(env_fn, env=StarCraft2Env2Wrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH", os.path.join(home_dir, "StarCraftII"))

# Pretty print supported environments


def print_supported_environments():
    print(Fore.GREEN + Style.BRIGHT + "Supported Environments:")
    for env_name, env_fn in REGISTRY.items():
        print(f"{Fore.CYAN}{Style.BRIGHT}{env_name}{Style.RESET_ALL}: {Fore.YELLOW}Available{Style.RESET_ALL}")


print_supported_environments()
