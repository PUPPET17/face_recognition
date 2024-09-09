import dlib
print(dlib.DLIB_USE_CUDA)  # 应该输出 True
print(dlib.cuda.get_num_devices())  # 应该输出 1 或更多，表示可用的 GPU 数量