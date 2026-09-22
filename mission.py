"""Выходы уровня из LINEDEFS WAD; прохождение только игровыми кнопками."""
import math
import struct
from pathlib import Path


def map_data(wad, name):
    data=Path(wad).read_bytes()
    count,offset=struct.unpack_from('<ii',data,4)
    directory=[struct.unpack_from('<ii8s',data,offset+i*16) for i in range(count)]
    index=next(i for i,(_,_,n) in enumerate(directory) if n.rstrip(b'\0').decode()==name.upper())
    lumps={n.rstrip(b'\0').decode():data[p:p+size] for p,size,n in directory[index+1:index+11]}
    vertices=list(struct.iter_unpack('<hh',lumps['VERTEXES']))
    exits=[]
    sides=list(struct.iter_unpack('<hh8s8s8sH',lumps['SIDEDEFS']))
    door_sectors=set()
    for index,line in enumerate(struct.iter_unpack('<7H',lumps['LINEDEFS'])):
        start,end,flags,special,tag,front,back=line
        if special in (1,26,27,28,31,32,33,34,117,118) and back!=65535:
            door_sectors.add(sides[back][-1])
        if special not in (11,51,52,124):continue
        a,b=vertices[start],vertices[end]
        center=((a[0]+b[0])/2,(a[1]+b[1])/2)
        length=math.dist(a,b)
        normal=((b[1]-a[1])/length,-(b[0]-a[0])/length)
        exits.append({'line':index,'special':special,'center':center,
                      'approach':(center[0]+normal[0]*40,center[1]+normal[1]*40),
                      'use':special in (11,51),'secret':special in (51,124)})
    return {'name':name.upper(),'exits':exits,'door_sectors':sorted(door_sectors)}


class Mission:
    def __init__(self,data):
        self.data=data
        self.exit=next((e for e in data['exits'] if not e['secret']),None)

    def objective(self):
        return self.exit['approach'] if self.exit else None

    def activate(self,s,tick):
        if not self.exit:return None
        center=self.exit['center']
        if math.dist((s['x'],s['y']),center)>60:return None
        angle=math.degrees(math.atan2(center[1]-s['y'],center[0]-s['x']))
        turn=-((angle-s['angle']+180)%360-180)
        return [float(not self.exit['use'] and abs(turn)<15),0,0,0,
                max(-6,min(6,turn)),0,float(self.exit['use'] and abs(turn)<8 and tick%12==0)]
