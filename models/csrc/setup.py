from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules():
    return [
        CUDAExtension(
            name='_msmv_sampling_cuda',
            sources=[
                'msmv_sampling/msmv_sampling.cpp',
                'msmv_sampling/msmv_sampling_forward.cu',
                'msmv_sampling/msmv_sampling_backward.cu'
            ],
            include_dirs=['msmv_sampling']
        )
    ]


setup(
    name='csrc',
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': BuildExtension}
)

