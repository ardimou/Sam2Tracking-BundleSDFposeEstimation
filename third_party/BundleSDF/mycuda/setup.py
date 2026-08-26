from setuptools import setup
import os

# Import torch only when needed to avoid build isolation issues
def get_extensions():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        raise RuntimeError("PyTorch is required to build this package. Install with: pip install torch")
    
    nvcc_flags = ['-Xcompiler', '-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
    c_flags = ['-O3', '-std=c++17']
    
    ext_modules = [
        CUDAExtension('common', [
            'bindings.cpp',
            'common.cu',
        ], extra_compile_args={'gcc': c_flags, 'nvcc': nvcc_flags}),
        CUDAExtension('gridencoder', [
            'torch_ngp_grid_encoder/gridencoder.cu',
            'torch_ngp_grid_encoder/bindings.cpp',
        ], extra_compile_args={'gcc': c_flags, 'nvcc': nvcc_flags}),
    ]
    
    return ext_modules, BuildExtension

# Build extensions
ext_modules, BuildExt = get_extensions()

setup(
    name='common',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    include_dirs=[
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ],
    zip_safe=False,
    # Add build requirements
    setup_requires=['torch>=2.6.0'],
    install_requires=['torch>=2.6.0'],
)
