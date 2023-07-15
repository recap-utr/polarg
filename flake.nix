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
    poetry2nix,
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
        python = pkgs.python311;
        poetry = pkgs.poetry;
        upload = pkgs.writeShellApplication {
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
              wi2gpu:recap-utr/argument-nli
          '';
        };
      in {
        packages = {
          default = poetry2nix.legacyPackages.${system}.mkPoetryApplication {
            inherit python;
            projectDir = ./.;
            preferWheels = true;
          };
        };
        devShells.default = pkgs.mkShell {
          packages = [poetry python upload];
          POETRY_VIRTUALENVS_IN_PROJECT = true;
          LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [stdenv.cc.cc zlib "/run/opengl-driver"];
          # TODO: https://github.com/google/sentencepiece/issues/889
          PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python";
          shellHook = ''
            ${lib.getExe poetry} env use ${lib.getExe python}
            ${lib.getExe poetry} install --all-extras --no-root
          '';
        };
      };
    };
}
