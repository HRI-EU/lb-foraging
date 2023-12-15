#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Utility functions for LBF experiments.
#
# The MIT License (MIT)
#
# Copyright © 2023 Honda Research Institute Europe GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
import sys
import inspect
import logging
import copy
from pathlib import Path

import yaml


def read_settings(filename):
    """Read settings from JSON or YAML configuration files.

    Reads settings file (JSON, YAML) from disk and return settings dictionary.

    Parameters
    ----------
    filename : str
        Filename of settings file

    Returns
    -------
    DotDict
        Settings dictionary
    """
    with open(filename, "r") as read_file:
        settings = yaml.safe_load(read_file)
    return DotDict(settings)


def initialize_logger(debug, logdir=None, logname=None, filemode="w"):
    """Set up root logger and initialize logging.

    Configure the root logger such that logs are printed to console and a log
    file in a subfolder. Also set format of logging output.

    Parameters
    ----------
    debug : bool
        If true, set logging level to logging.DEBUG, otherwise level is
        logging.INFO.
    logdir : str | None, optional
        Directory to which logging files a written (in a dubdirectory 'log'),
        if None the log subdirectory is created in the current working
        directory, by default None
    logname : str | None, optional
        Name of logfile, if None logname is set to the calling modules name, by
        default None
    filemode : str, optional
        Whether to append to existing log ('a') or write a new one ('w'), by
        default 'w'
    """
    # Set log format
    log_format = "%(asctime)s - %(levelname)-4s  [%(filename)s:%(funcName)10s():l %(lineno)d] %(message)s"
    log_fmt = "%Y-%m-%d - %H:%M:%S"
    if debug:
        logging.basicConfig(
            format=log_format, datefmt=log_fmt, level=logging.DEBUG
        )
    else:
        logging.basicConfig(
            stream=sys.stdout,
            format=log_format,
            datefmt=log_fmt,
            level=logging.INFO,
        )

    # Create a folder to store all logs. If no logdir is provided write to the
    # current working directory.
    if logdir is None:
        logdir = Path.cwd()
    else:
        if not Path(logdir).exists():
            raise RuntimeError(f"logdir {logdir} does not exist!")
    Path(logdir).joinpath("log").mkdir(parents=True, exist_ok=True)

    # Add a handler that writes info messages to a log of the same name as the
    # analysis file calling this function. Handle special cases: Windows OS and Pycharm runs
    # return the full path of the module while Linux doesn't (called from bash); identify calls
    # from the iPython shell and trim numbers from module name.
    if logname is None:
        module_name = inspect.stack()[1].filename.split(".")[0]
        if module_name.find("\\"):  # to run this in iPython under Windows
            module_name = module_name.split("\\")[-1]
        if "/" in module_name:  # to run this from Pycharm
            module_name = module_name.split("/")[-1]
        if "ipython-input" in module_name:
            logname = "ipython-input"
            logging.info(  # pylint: disable=W1201
                "Logger is set from iPython session, logging to %s*.log ."  # pylint: disable=C0209
                % logname
            )
        else:
            logname = module_name
    if debug:
        logname = f"{logname}_debug"
    fh = logging.FileHandler(
        Path(logdir).joinpath("log", f"{logname}.log"), mode=filemode
    )
    if debug:
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger("").addHandler(fh)  # add to root logger


class DotDict(dict):
    """Dictionary with dot-notation access to values.

    Provides the same functionality as a regular dict, but also allows
    accessing values using dot-notation.
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        """Return dictionary keys as list of attributes."""
        return self.keys()

    def __deepcopy__(self, memo):
        """Provide deep copy capabilities.

        Following a fix described here:
        https://github.com/aparo/pyes/pull/115/commits/d2076b385c38d6d00cebfe0df7b0d1ba8df934bc
        """
        dot_dict_copy = DotDict(
            [
                (copy.deepcopy(k, memo), copy.deepcopy(v, memo))
                for k, v in self.items()
            ]
        )
        return dot_dict_copy

    def __getstate__(self):
        # For pickling the object
        return self

    def __setstate__(self, state):
        # For un-pickling the object
        self.update(state)
        # self.__dict__ = self
