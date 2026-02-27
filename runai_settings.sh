# For Meteo-France's users of runai, to avoid "Permissions denied" warning
export RUNAI_EXTRA_MOUNTS="-v $HOME/.config/earthkit:$HOME/.config/earthkit"