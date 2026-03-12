# For Meteo-France's users of runai:
# -v $HOME/.config/earthkit:$HOME/.config/earthkit -> to avoid "Permissions denied" warning.
# -e SCIPY_ARRAY_API=1 -> enables array API from `scipy` (https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html).
export RUNAI_EXTRA_MOUNTS="-v $HOME/.config/earthkit:$HOME/.config/earthkit -e SCIPY_ARRAY_API=1"
