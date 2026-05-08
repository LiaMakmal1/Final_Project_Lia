# setup.py -- Build the symnmfmodule Python C extension.
#
# Run with:
#   python3 setup.py build_ext --inplace
#
# This produces symnmfmodule.<platform>.so in the current directory,
# which allows symnmf.py and analysis.py to import symnmfmodule directly.

from setuptools import Extension, setup

module = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"],
    # SYMNMF_PYTHON_MODULE suppresses the standalone main() in symnmf.c
    define_macros=[("SYMNMF_PYTHON_MODULE", "1")],
    extra_compile_args=["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic-errors"],
    extra_link_args=[],
    libraries=["m"],
)

setup(
    name="symnmfmodule",
    version="1.0",
    ext_modules=[module],
)
