from setuptools import Extension, setup

module = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"],
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