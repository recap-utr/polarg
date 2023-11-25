# PolArg: Unsupervised Polarity Prediction of Arguments

This application consists of a client and a server.

## Server Usage

```shell
SERVER_ADDRESS="127.0.0.1:50100"
nix develop -c poetry run python -m polarg.deploy "$SERVER_ADDRESS"
```

## Evaluation Client

```shell
nix develop -c poetry run python -m polarg.evaluate "$SERVER_ADDRESS" "$TEST_FOLDER" "$TEST_FOLDER_GLOB_PATTERN"
```

The evaluation has multiple parameters, view them as follows:

```shell
nix develop -c poetry run python -m polarg.evaluate --help
```
