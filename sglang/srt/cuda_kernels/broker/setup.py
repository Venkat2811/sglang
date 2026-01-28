from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='broker_moe_kernel',
    ext_modules=[
        CUDAExtension(
            name='broker_moe_kernel',
            sources=['broker_moe_dispatch.cu'],
            include_dirs=['.'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
