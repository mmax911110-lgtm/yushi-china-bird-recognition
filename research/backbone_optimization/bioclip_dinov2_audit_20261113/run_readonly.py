"""Reuse the installed runtime. Keep all audit output/cache in this directory."""
from pathlib import Path
import os,runpy,sys
HERE=Path(__file__).resolve().parent
PACKAGE=Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA')
runtime=PACKAGE/'runtime'
sys.path[:]=[str(HERE),str(PACKAGE/'app')]+[str(runtime/'packages'/name) for name in ('primary','cuda','base')]+[
    p for p in sys.path if p and Path(p).resolve().is_relative_to(runtime.resolve())]
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
for key in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','YOLO_CONFIG_DIR'):
    directory=HERE/'cache'/key;directory.mkdir(parents=True,exist_ok=True);os.environ[key]=str(directory)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false')
handles=[os.add_dll_directory(str(p)) for p in (runtime,runtime/'DLLs',runtime/'packages/cuda/torch/lib')]
script=Path(sys.argv[1]).resolve();sys.argv=[str(script)]+sys.argv[2:]
runpy.run_path(str(script),run_name='__main__')
