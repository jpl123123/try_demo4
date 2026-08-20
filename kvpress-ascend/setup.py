"""setuptools build for kvpress-ascend.

The wheel must carry ``kvpress_ascend.pth`` at the *wheel root* so that pip
installs it into site-packages and every Python interpreter startup runs
``import kvpress_ascend`` (which is env-gated and lazy).
"""
import os

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.abspath(os.path.dirname(__file__))
PTH_SRC = os.path.join(HERE, "kvpress_ascend.pth")


class build_py_with_pth(build_py):
    """Copy the .pth file into the build root (== wheel root)."""

    def run(self) -> None:
        super().run()
        target = os.path.join(self.build_lib, "kvpress_ascend.pth")
        self.mkpath(os.path.dirname(target))
        with open(PTH_SRC, "r", encoding="utf-8") as fsrc:
            content = fsrc.read()
        with open(target, "w", encoding="utf-8") as fdst:
            fdst.write(content)


setup(
    name="kvpress-ascend",
    version="0.1.0",
    description=(
        "Monkeypatch adapter: kvpress KV-cache compression for vllm-ascend "
        "v0.23.0 (zero changes to vllm-ascend source). Enable with "
        "export kvpress=1 before starting vllm serve."
    ),
    long_description=open(os.path.join(HERE, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=["kvpress_ascend", "kvpress_ascend.runtime"],
    cmdclass={"build_py": build_py_with_pth},
    zip_safe=False,
)
