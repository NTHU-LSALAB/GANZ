# GPU Packet Processing - Build Configuration
# SPDX-License-Identifier: BSD-3-Clause

if not flag_enable_gpu_support
	warning('Skipping compilation of GAZSI - Missing GPU support.')
	subdir_done()
endif

app_dependencies += declare_dependency(link_args : '-lresolv')
app_dependencies += gpu_dependencies

doca_gpu_dep = dependency('doca-gpunetio')
gpunetio_device_path = doca_gpu_dep.get_variable(pkgconfig : 'libdir')
dependency_gpunetio_device = declare_dependency(compile_args : '-Wl,--whole-archive',
                                                link_args : ['-L' + gpunetio_device_path , '-ldoca_gpunetio_device'],)
app_dependencies += doca_gpu_dep
app_dependencies += dependency_gpunetio_device

# Add TensorRT dependencies
tensorrt_dep = declare_dependency(
	compile_args : ['-I/usr/include/x86_64-linux-gnu'],
	link_args : ['-lnvinfer', '-lnvinfer_plugin']
)
app_dependencies += tensorrt_dep

# DOCA initialization and queue management
app_srcs = files([
	'doca/args.c',
	'doca/device.c',
	'doca/flow.c',
	'doca/http_tx.c',
	'doca/tcp.c',
])

# TCP session management
app_srcs += files([
	'tcp/cpu_rss.c',
	'tcp/session.c',
])

# Utility functions
app_srcs += files([
	'utils/utils.c',
])

# GPU kernels
app_srcs += files([
	'kernels/tcp_rx.cu',
	'kernels/http_server.cu',
])

# Inference implementation
app_srcs += files([
	'inference/gpu_pipeline.cu',
	'inference/ring_buffer.cu',
	'inference/tensorrt.cu',
])

vanilla_app_srcs = [
	'main.c',
]

app_inc_dirs += include_directories('doca')
app_inc_dirs += include_directories('utils')
app_inc_dirs += include_directories('tcp')
app_inc_dirs += include_directories('inference')
app_inc_dirs += include_directories('kernels')

executable('GAZSI',
		app_srcs + vanilla_app_srcs,
		c_args : gpu_c_args,
		cuda_args : gpu_cuda_args + ['-std=c++17'],
		dependencies : app_dependencies,
		include_directories : app_inc_dirs,
		install: install_apps)
