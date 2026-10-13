{
  description = "Basic Python development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          git
          ruff
        ];

        shellHook = ''
          set -a && source .env && set +a

          export UV_PYTHON_DOWNLOADS=never
          export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          echo "Python development environment"
          python --version
          uv --version
        '';
      };
    };
}
