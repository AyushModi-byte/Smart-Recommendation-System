"""Minimal torch-free reader for a state_dict saved with torch.save (zip format). numpy only."""
import pickle, zipfile, numpy as np
from collections import OrderedDict

_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
       "HalfStorage": np.float16, "IntStorage": np.int32}

class _Storage:
    def __init__(self, dtype): self.dtype = dtype

def _rebuild(storage, offset, size, stride, *a, **k):
    arr = storage.data
    n = int(np.prod(size)) if len(size) else 1
    flat = arr[offset:offset + n] if offset or n != arr.size else arr
    return np.lib.stride_tricks.as_strided(
        flat, shape=tuple(size), strides=tuple(s * flat.itemsize for s in stride)).copy()

def load_state_dict(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2": return _rebuild
            if name in _DT: return name
            if name == "OrderedDict": return OrderedDict
            if mod.startswith("torch") and name == "_utils": return None
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, st, key, _loc, _n = pid
            s = _Storage(_DT[st if isinstance(st, str) else st.__name__])
            s.data = np.frombuffer(z.read(f"{prefix}/data/{key}"), dtype=s.dtype)
            return s
    return U(z.open(f"{prefix}/data.pkl")).load()
