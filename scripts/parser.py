import yaml
import pathlib

_AUTHORIZED_KEYS = {
    'preprocessing': [
        'beamline',
        'binning',
        'rocking-angle',
        'data-root-folder',
        'debug',
        'detector',
        'filter',
        'interact',
        'mask',
        'orthogonalize',
        'output-dir',
        'sample-name',
        'save-dir',
        'sample_detector_distance',
        'scan',
        'specfile-path',
        'template-data-file'
    ],
    'postprocessing': [
        'angle-step',
        'beamline',
        'data-root-folderscan',
        'debug',
        'energy',
        'flip',
        'incidence-angle',
        'inplane-angle',
        'is-orthogonalized',
        'isosurface-threshold',
        'modes',
        'outofplane-angle',
        'rocking-angle',
        'rocking-angle',
        'sample-detector-distance',
        'save-dir',
        'specfile-path',
        'voxel-size'
    ]
}


# typo: ArgumentHandler, ArgumentParser, ArgumentLoader, Argument, Parser ??
class ArgumentHandler():
    """
    Base class to deal with arguments required by scripts

    :param file_path: path of the configuration file that contains
    the arguments
    """
    def __init__(self, file_path, script_type="preprocessing"):
        self.file_path = file_path
        if script_type not in _AUTHORIZED_KEYS.keys():
            print("Please, provide a script_type from "
                  f"{_AUTHORIZED_KEYS.keys()}")
        else:
            self.script_type = script_type

        self.raw_config = self._open_file()
        self.arguments = None

    def _open_file(self):
        """open the file and return it"""
        with open(self.file_path, "rb") as f:
            raw_config = f.read()
        return raw_config

    def load_arguments(self):
        extension = self._get_extension()
        if extension == ".yml":
            args = yaml.load(self.raw_config, Loader=yaml.FullLoader)
            self.arguments = self._checked_args(args)
            return self.arguments
        else:
            return None

    def _get_extension(self):
        """return the extension of the the file_path attribute"""
        return pathlib.Path(self.file_path).suffix

    def _checked_args(self, dic):
        checked_keys = []
        for key in dic.keys():
            if key not in _AUTHORIZED_KEYS[self.script_type]:
                print(f"'{key}' is an unexpected key, "
                      "its value won't be considered.")
            else:
                checked_keys.append(key)
        return {key: dic[key] for key in checked_keys}

    # For now the yam Loader already returns a dic, so not useful
    # but we may need it if we use other file format
    def to_dict(self):
        pass

    def dump(self, output_path, extension):
        pass


if __name__ == '__main__':
    config_file = "../conf/default_config.yml"
    arg_handler = ArgumentHandler(
        config_file,
        script_type="postprocessing"  # try with "postprocessing"
        )

    args = arg_handler.load_arguments()  # this can also be accessed by
    # arg_handler.arguments once load_arguments() has been computed

    print(f"The current configuration file is:\n{config_file}\n")
    print("attribute arg_handler.arguments:")
    print(arg_handler.arguments)  # or args
