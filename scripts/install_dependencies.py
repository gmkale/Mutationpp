#!/usr/bin/python3

import argparse
import multiprocessing
import os
import subprocess

from utils import colors, reindent


class Dependency(object):
    """Default installation method for packages using CMake as their build system"""

    def __init__(self, name, build_cmd_seq, build_dir=None, source_dir=None):
        self.name = name
        self.source_dir = os.path.join(THIRDPARTY_DIR, self.name) if source_dir is None else source_dir
        self.build_dir = os.path.join(self.source_dir, "build") if build_dir is None else build_dir
        self.build_cmd_seq = build_cmd_seq

    def install(self):

        print(colors.fg.red + "Installing " + self.name + colors.reset)
        os.chdir(self.source_dir)
        os.makedirs(self.build_dir, exist_ok=True)  # This is why Python3 is required
        os.chdir(self.build_dir)

        for cmd in self.build_cmd_seq:
            print(colors.fg.blue + " ".join(cmd) + colors.reset)
            subprocess.call(cmd)


# Global variables
# TODO: check if environment variables are defined
THIRDPARTY_DIR = os.path.join(os.environ["MPP_DIR"], "thirdparty")
INSTALL_DIR = os.path.join(os.environ["MPP_DIR"], "install")
NUM_PROC = multiprocessing.cpu_count()

DEPS_DICT = \
    {
        "eigen": Dependency(name="eigen",
                            build_cmd_seq=[["cmake", "-DCMAKE_INSTALL_PREFIX={}".format(INSTALL_DIR), ".."],
                                           ["make", "-j", str(NUM_PROC), "install"]]),
        "catch": Dependency(name="catch",
                            build_cmd_seq=[["cmake", "-DCMAKE_INSTALL_PREFIX={}".format(INSTALL_DIR), ".."],
                                           ["make", "-j", str(NUM_PROC), "install"]]),
    }


def main(args):

    deps_list = sorted(DEPS_DICT.keys())

    if args.list:
        print("Dependencies available for installation:\n" +
              reindent("\n".join(deps_list), 4))
        return

    if not args.deps:  # Means that no positional argument is provided
        for name in deps_list:
            DEPS_DICT[name].install()
    else:
        for name in args.deps:
            DEPS_DICT[name].install()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Install third-party dependencies for the Mutation++ software package.""")
    parser.add_argument("-l", "--list", action="store_true",
                        help="show the list of available dependencies")
    parser.add_argument('deps', nargs='*', default=None,
                        help="list of dependencies to install (if none is specified, install all dependencies)")
    args = parser.parse_args()

    main(args)
