import typing as t

from dynaconf import Dynaconf

config = t.cast(
    t.Any,
    Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=["settings.toml", ".secrets.toml"],
    ),
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
