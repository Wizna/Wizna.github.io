![title image](https://picjumbo.com/wp-content/uploads/beautiful-green-field-scenery-2210x1473.jpg)

### Background

Sometimes we may run into annoying bugs or problems, and a hint on what's probably going wrong is valuable.

### TypeError: 'module' object is not callable

This error may shows up when you use pip on the command line, e.g. `pip show numpy`.

This error maybe result from a installation of pip to the system level, then `install` or `upgrade` pip with --user argument to user level.

#### To fix

run `python -m pip uninstall pip` to uninstall current pip, then we will fall back to the old pip on system level.

### ModuleNotFoundError: No module named '_pydevd_bundle.pydevd_cython'

This error may happen when using PyCharm debugger. The reason is probably that you have a python package with a name conflicts with that of debugger, e.g. "code". 

#### To fix

rename your personal python packages.

### To be continued