from torch.utils.cpp_extension import load

y2 = load(name='y2', sources=['y2_compress_decompress.cpp', 'y2_compress_decompress.cu'])