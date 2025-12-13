from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import subprocess

# Custom CMake build helper
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # cmake_args for PyBind11
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Ensure single-config generators (e.g. Makefiles/Ninja) get a build type.
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Allow passing extra CMake args via env var, e.g.:
        # CMAKE_ARGS="-DNF_USE_FETCHCONTENT=OFF -DNF_UWEBSOCKETS_DIR=... -DNF_USOCKETS_DIR=..." pip install -e backend_extension
        extra = os.environ.get("CMAKE_ARGS", "")
        if extra:
            cmake_args += extra.split()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Config
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        
        # Build
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='synapse',
    version='0.1.0',
    description='High-performance neural network activation probe',
    ext_modules=[CMakeExtension('synapse')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
