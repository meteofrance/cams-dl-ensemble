# export ECCODES_DEFINITION_PATH="/scratch/shared/def_eccodes/local_grib"
export RUNAI_EXTRA_MOUNTS="-v $HOME/.ecmwfdatastoresrc:$HOME/.ecmwfdatastoresrc -e ECCODES_DEFINITION_PATH -v $HOME/.config/earthkit/:$HOME/.config/earthkit/"
