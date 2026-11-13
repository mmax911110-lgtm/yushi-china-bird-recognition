"""Read-only structural audit. No model loading, training, or downloads."""
from pathlib import Path
from collections import Counter
import ast
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import tokenize

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
DATA=WORK/'bioclip2_china_eval_20261028'
PACKAGE=Path(r'D:\项目\鸟类识别\识别模型\v1\v1.1\NVIDIA')
DATASETS=Path(r'D:\项目\鸟类识别\识别模型\数据集')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):digest.update(part)
    return digest.hexdigest()
def save(name,value):
    with (HERE/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def tree(root,depth=2):
    result=[]
    def walk(folder,level):
        try:entries=sorted(folder.iterdir())
        except OSError as e:
            result.append({'path':str(folder),'error':str(e)});return
        for path in entries:
            if not path.is_dir():continue
            result.append({'path':str(path),'resolved':str(path.resolve()),'junction':path.resolve()!=path.absolute()})
            if level<depth:walk(path,level+1)
    walk(root,1)
    return result


def safetensor_header(path):
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        assert length<16*1024**2
        header=json.loads(stream.read(length))
    values={k:v for k,v in header.items() if k!='__metadata__'}
    visual={k:v for k,v in values.items() if k.startswith('visual.')}
    return {'path':str(path),'bytes':path.stat().st_size,'metadata':header.get('__metadata__'),
            'tensor_count':len(values),'visual_tensor_count':len(visual),
            'visual_stored_elements':sum(math.prod(v['shape']) for v in visual.values()),
            'all_stored_elements':sum(math.prod(v['shape']) for v in values.values()),
            'visual_dtypes':dict(Counter(v['dtype'] for v in visual.values())),
            'visual_shapes':{k:v['shape'] for k,v in visual.items()},
            'weights_loaded_into_model':False}


def inventory():
    prune={'.venv','runtime','site-packages','packages','node_modules','sources','data','cache',
           'features','evaluation','scoring','predictions','fresh_data','__pycache__','.git'}
    roots=[DATA,PACKAGE/'app',WORK/'adaptive_fusion_20261113',WORK/'nonlinear_fusion_20261113',WORK/'amu_fusion_20261113']
    result=[]
    for root in roots:
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in prune]
            for file in sorted(files):
                if not file.endswith('.py'):continue
                path=Path(folder)/file
                with tokenize.open(path) as stream:text=stream.read()
                row={'path':str(path),'resolved':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path),'lines':len(text.splitlines())}
                try:
                    parsed=ast.parse(text)
                    row['classes']=[n.name for n in parsed.body if isinstance(n,ast.ClassDef)]
                    row['functions']=[n.name for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                    row['imports']=sorted({(n.module or '') if isinstance(n,ast.ImportFrom) else a.name
                                           for n in ast.walk(parsed) if isinstance(n,(ast.Import,ast.ImportFrom))
                                           for a in (n.names if isinstance(n,ast.Import) else [None])})
                except SyntaxError as e:row['parse_error']=str(e)
                result.append(row)
    save('source_inventory.json',result)
    print('SOURCE_FILES',len(result),'LINES',sum(r['lines'] for r in result),flush=True)


def main():
    save('dataset_directory_map.json',tree(DATASETS,3))
    cfg=read(PACKAGE/'configs/recognizer.json')
    save('deployed_configuration.json',cfg)
    assets={}
    for relative,expected in cfg['asset_hashes'].items():
        path=PACKAGE/relative
        actual=sha(path)
        assert actual==expected,relative
        assets[relative]={'path':str(path),'bytes':path.stat().st_size,'sha256':actual,'matches_deployed_manifest':True}
    save('protected_assets_before.json',assets)
    models={kind:{'config':read(PACKAGE/f'models/{kind}/open_clip_config.json'),
                   'checkpoint':safetensor_header(PACKAGE/f'models/{kind}/open_clip_model.safetensors')}
            for kind in ['bioclip2','bioclip25']}
    save('model_headers.json',models)
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','available_extended']]
    memory=Memory();memory.length=ctypes.sizeof(Memory)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],capture_output=True,text=True)
    environment={'platform':platform.platform(),'python_for_audit':platform.python_version(),
                 'runtime_python':str(PACKAGE/'runtime/python.exe'),
                 'memory_total':memory.total_physical,'memory_available':memory.available_physical,
                 'disk':{drive:dict(zip(['total','used','free'],shutil.disk_usage(drive))) for drive in ['C:/','D:/']},
                 'gpu_query':gpu.stdout.strip(),'no_GPU_model_loaded':True}
    save('environment_system.json',environment)
    inventory()
    print('MODEL_STORED_ELEMENTS',{k:v['checkpoint']['visual_stored_elements'] for k,v in models.items()},flush=True)


if __name__=='__main__':main()
