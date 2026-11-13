from copy import deepcopy
from types import SimpleNamespace
import hashlib,json,math

TILE_SIZE=640
TILE_OVERLAP=.25
MAX_TILES=256

def tile_windows(size,tile_size=TILE_SIZE,overlap=TILE_OVERLAP,max_tiles=MAX_TILES):
    width,height=map(int,size)
    if width<1 or height<1 or tile_size<2 or not 0<=overlap<1:raise ValueError('无效分块尺寸或重叠率')
    if width<=tile_size and height<=tile_size:return []
    stride=max(1,int(tile_size*(1-overlap)))
    def starts(length):
        if length<=tile_size:return [0]
        values=list(range(0,length-tile_size+1,stride))
        if values[-1]!=length-tile_size:values.append(length-tile_size)
        return values
    xs,ys=starts(width),starts(height)
    count=len(xs)*len(ys)
    if count>max_tiles:
        raise ValueError(f'此图片需要 {count} 个分块，超过 {max_tiles} 个上限。请先裁出需要检查的区域；本次未进行部分扫描。')
    return [[x,y,min(width,x+tile_size),min(height,y+tile_size)] for y in ys for x in xs]

def map_tile_box(local_box,window,image_size,tolerance=1.):
    width,height=image_size;left,top,right,bottom=window
    if len(local_box)!=4 or not all(math.isfinite(float(v)) for v in local_box):return None,['invalid_box']
    x1,y1,x2,y2=map(float,local_box);tw,th=right-left,bottom-top
    if x2<=x1 or y2<=y1 or x1<0 or y1<0 or x2>tw or y2>th:return None,['invalid_box']
    cut=[]
    if left>0 and x1<=tolerance:cut.append('left')
    if top>0 and y1<=tolerance:cut.append('top')
    if right<width and x2>=tw-tolerance:cut.append('right')
    if bottom<height and y2>=th-tolerance:cut.append('bottom')
    return [x1+left,y1+top,x2+left,y2+top],cut

def image_identity(image):
    image=image.convert('RGB')
    return {'width':image.width,'height':image.height,'mode':'RGB',
        'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest()}

def packet_digest(packet):
    body={k:v for k,v in packet.items() if k!='packet_sha256'}
    return hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

def make_initial_packet(image,ordinary,model_hashes,configuration,elapsed_ms,device):
    packet={'schema':'bird-initial-v1','image':image_identity(image),
        'detector':{'model_hashes':deepcopy(model_hashes),'configuration':deepcopy(configuration)},
        'ordinary':{'boxes':deepcopy(ordinary.boxes),'passes':ordinary.passes,
            'detector_candidates':ordinary.detector_candidates,'filtered_candidates':ordinary.filtered_candidates},
        'elapsed_ms':float(elapsed_ms),'source_device':device}
    packet['packet_sha256']=packet_digest(packet)
    return packet

def validate_initial_packet(packet,image,model_hashes,configuration):
    message='初检数据与当前图片或冻结检测模型不匹配，请对原图重新初检'
    try:
        if not isinstance(packet,dict) or set(packet)!={'schema','image','detector','ordinary','elapsed_ms','source_device','packet_sha256'}:raise ValueError(message)
        if packet['schema']!='bird-initial-v1' or packet['packet_sha256']!=packet_digest(packet):raise ValueError(message)
        if packet['image']!=image_identity(image):raise ValueError(message)
        if packet['detector']!={'model_hashes':model_hashes,'configuration':configuration}:raise ValueError(message)
        if packet['source_device'] not in ('cpu','cuda'):raise ValueError(message)
        elapsed=packet['elapsed_ms']
        if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:raise ValueError(message)
        ordinary=packet['ordinary']
        if set(ordinary)!={'boxes','passes','detector_candidates','filtered_candidates'}:raise ValueError(message)
        if type(ordinary['passes']) is not int or ordinary['passes']!=1:raise ValueError(message)
        if not isinstance(ordinary['boxes'],list):raise ValueError(message)
        for key in ('detector_candidates','filtered_candidates'):
            if type(ordinary[key]) is not int or ordinary[key]<0:raise ValueError(message)
        if ordinary['detector_candidates']!=len(ordinary['boxes'])+ordinary['filtered_candidates']:raise ValueError(message)
        for box in ordinary['boxes']:
            if set(box)!={'xyxy','original_xyxy','confidence','class_id','class_name'}:raise ValueError(message)
            if type(box['class_id']) is not int or box['class_id'] not in configuration['classes']:raise ValueError(message)
            if not isinstance(box['class_name'],str) or not box['class_name']:raise ValueError(message)
            confidence=box['confidence']
            if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not configuration['conf']<=confidence<=1:raise ValueError(message)
            bounds=box['xyxy']
            if not isinstance(bounds,list) or len(bounds)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in bounds):raise ValueError(message)
            x1,y1,x2,y2=bounds
            if not 0<=x1<x2<=image.width or not 0<=y1<y2<=image.height or box['original_xyxy']!=bounds:raise ValueError(message)
        return SimpleNamespace(**deepcopy(ordinary))
    except (TypeError,KeyError,OverflowError) as error:
        raise ValueError(message) from error
