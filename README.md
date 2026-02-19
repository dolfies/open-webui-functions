# Open WebUI Functions

Contains the functions I use in my own Open WebUI setup. Many are derivatives of existing functions that I customized to my needs, and some are original functions that I created from scratch. Anything I created myself is licensed under the MIT License, and anything that is a derivative of existing functions is licensed under the same license as the original function.

The purpose of this combination of functions is to facilitate an Open WebUI setup that supports all the major AI providers, as well as all their major features, in a cohesive, user-friendly way. Having everything available in a single setup allows for easy comparison and switching between providers, as well as the ability to use multiple providers in a single workflow. This is far from perfect as Open WebUI is not as flexible as I would like it to be, but it's pretty good for now and I will continue to improve it as I go.

## Management

The `manage.py` script provides a command-line interface for managing Open WebUI functions. It allows you to extract functions from a JSON export and bundle local files into a JSON export for importing into Open WebUI.

```sh
# Extract functions from a JSON export
python manage.py extract path/to/export.json

# Bundle local files into a JSON export
python manage.py bundle output.json
```

In the future, I would like to hack Open WebUI to get it pulling functions directly from the repository, but for now this will have to do.

## Contribution

This repository is not really meant for public contribution, as it contains a lot of personal customizations and is not really organized in a way that would make it easy for others to contribute (not to even mention that I manually pull upstream changes for some of these functions). However, if you have any suggestions or improvements, feel free to open an issue or a pull request.
