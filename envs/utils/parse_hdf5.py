import h5py
import numpy as np

from .rgb_codec import decode_hdf5_rgb, require_current_rgb_jpeg_codec


def parse_img_array(data):
    """
    将一个字节流数组解码为图像数组。

    Args:
        data: np.ndarray of shape (N,), 每个元素要么是 Python bytes，要么是 np.ndarray(dtype=uint8)
    Returns:
        imgs: np.ndarray of shape (N, H, W, C), dtype=uint8
    """
    # 确保 data 是可迭代的一维数组
    flat = data.ravel()

    imgs = []
    for buf in flat:
        # buf 可能是 bytes，也可能是 np.ndarray(dtype=uint8)
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported buffer type: {type(buf)}")

        img = decode_hdf5_rgb(arr)
        imgs.append(img)

    # 将 list 转成形如 (N, H, W, C) 的 ndarray
    return np.stack(imgs, axis=0)


def h5_to_dict(node):
    result = {}
    for name, item in node.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if "rgb" in name:
                result[name] = parse_img_array(data)
            else:
                result[name] = data
        elif isinstance(item, h5py.Group):
            # 递归处理子 group
            result[name] = h5_to_dict(item)
    # 如果你还想把 attributes 一并读进来，可以：
    if hasattr(node, "attrs") and len(node.attrs) > 0:
        result["_attrs"] = dict(node.attrs)
    return result


def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        require_current_rgb_jpeg_codec(f)
        data_dict = h5_to_dict(f)
    return data_dict
