"""Script that train and validate a model.
Use cases:
- `fit_and_val.py --config config/path.yaml`

Default values for a config file can be displayed with:
```
runai python scripts/main.py fit --print_config
```
"""


def fit_and_val(args: list[str] = []) -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    fit_and_val()
