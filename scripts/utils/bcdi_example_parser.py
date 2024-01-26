#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Clément Atlan, c.atlan@outlook.com

from bcdi.utils.parser import ConfigParser

if __name__ == "__main__":
    config_file = "../../bcdi/examples/S11_config_preprocessing.yml"
    arg_handler = ConfigParser(config_file)

    args = arg_handler.load_arguments()  # this can also be accessed by
    # arg_handler.arguments once load_arguments() has been computed

    print(f"The current configuration file is:\n{config_file}\n")
    print("attribute arg_handler.arguments:")
    print(arg_handler.arguments)  # or args
