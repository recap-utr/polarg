{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = inputs @ {
    flake-parts,
    nixpkgs,
    systems,
    poetry2nix,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import systems;
      perSystem = {
        pkgs,
        lib,
        system,
        self',
        ...
      }: let
        python = pkgs.python311;
        poetry = pkgs.poetry;
      in {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          overlays = [poetry2nix.overlays.default];
        };
        packages = {
          default = pkgs.poetry2nix.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
          };
          upload = pkgs.writeShellApplication {
            name = "upload";
            text = ''
              exec ${lib.getExe pkgs.rsync} \
                --progress \
                --archive \
                --delete \
                --exclude "data/*.pt" \
                --include "data/*" \
                --include "/.env" \
                --exclude-from ".gitignore" \
                --exclude ".git" \
                ./ \
                wi2gpu:recap-utr/polarg
            '';
          };
        };
        devShells.default = pkgs.mkShell {
          packages = [poetry python self'.packages.upload];
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [stdenv.cc.cc zlib "/run/opengl-driver"];
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root --sync
            set -a
            source .env
          '';
        };
      };
    };
}
