from engforge.env_var import EnvVariable
import os

FORGE_ROOT_VAR = EnvVariable(
    "FORGE_ROOT", default=None, dontovrride=True, type_conv=os.path.expanduser
)


def client_path(alternate_path=None, **kw):
    path = FORGE_ROOT_VAR.secret
    if path is None:
        if alternate_path is None:
            raise KeyError(
                f"no `FORGE_ROOT` set and no alternate path in client_path call"
            )
        return alternate_path
    else:
        return path
