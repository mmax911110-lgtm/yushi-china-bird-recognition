"""Use the installed runtime and prior experiment without altering either."""
import os
from pathlib import Path
import runpy
import sys

here=Path(__file__).resolve().parent
runtime=Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA\runtime')
sys.path[:]=[str(here),str(here.parent/'dinov2_species_20261113'),str(runtime.parent/'app')]+[
    str(runtime/'packages'/part) for part in ('primary','cuda','base')]+[
    path for path in sys.path if path and Path(path).resolve().is_relative_to(runtime.resolve())]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='4'
for name in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','YOLO_CONFIG_DIR'):
    path=here/'cache'/name
    path.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(path)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true')
handles=[os.add_dll_directory(str(path)) for path in (runtime,runtime/'DLLs',runtime/'packages/cuda/torch/lib')]
script=(here/(sys.argv[1]+'.py')).resolve()
assert script.parent==here
sys.argv=[str(script),*sys.argv[2:]]
runpy.run_path(str(script),run_name='__main__')
