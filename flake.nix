{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };
  outputs = inputs @ {
    flake-parts,
    nixpkgs,
    systems,
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
        python = pkgs.python310;
        poetry = pkgs.poetry;
      in {
        apps.upload = {
          type = "app";
          program = lib.getExe (pkgs.writeShellApplication {
            name = "upload";
            text = ''
              exec ${lib.getExe pkgs.rsync} \
                --progress \
                --archive \
                --delete \
                --include-from ".rsyncinclude" \
                --exclude-from ".gitignore" \
                --exclude ".git" \
                ./ \
                wi2gpu:recap-utr/polarg
            '';
          });
        };
        packages = {
          default = pkgs.poetry2nix.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
          };
        };
        devShells.default = pkgs.mkShell {
          packages = [poetry python];
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [stdenv.cc.cc zlib "/run/opengl-driver"];
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root
            set -a
            source .env
          '';
        };
      };
    };
}
