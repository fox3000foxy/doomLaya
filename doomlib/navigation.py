"""Поиск проходимого маршрута по секторам ViZDoom с памятью посещённых областей."""
from collections import Counter
import heapq
import math

import numpy as np
from shapely import contains_xy, intersects_xy
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

STEP=16
REGION=256


class Navigator:
    def __init__(self,sectors=None):
        self.free=set()
        self.floors={}
        self.regions=Counter()
        self.entered=set()
        self.path=[]
        self.blocked=set()
        self.last_cell=None
        self.last_region=None
        self.last_new_tick=0
        self.goal_kind=None
        self.exhausted=False
        self.objective=None
        self.requested=None
        self.walk=None
        self.steering_target=None
        self.failed_request=None
        self.retry_tick=0
        if sectors is not None:self.configure(sectors)

    def configure(self,sectors):
        areas=[]
        records=[]
        walls=[]
        for sector in sectors:
            lines=[LineString([(l.x1,l.y1),(l.x2,l.y2)]) for l in sector.lines if (l.x1,l.y1)!=(l.x2,l.y2)]
            polygons=list(polygonize(lines))
            # Внутреннее кольцо другого сектора не превращаем в пол текущего.
            polygons=[p for p in polygons if not any(p is not q and Polygon(q.exterior).contains(p.representative_point()) for q in polygons)]
            if not polygons:continue
            area=unary_union(polygons)
            height=sector.ceiling_height-sector.floor_height
            door=height<=8 and len(sector.lines)<=8 and max(area.bounds[2]-area.bounds[0],area.bounds[3]-area.bounds[1])<=256
            if height>=56 or door:
                areas.append(area)
                records.append((area,float(sector.floor_height)))
            walls.extend(LineString([(l.x1,l.y1),(l.x2,l.y2)]) for l in sector.lines if l.is_blocking)
        walk=unary_union(areas).buffer(-18)
        if walls:walk=walk.difference(unary_union(walls).buffer(18))
        self.walk=walk
        xmin,ymin,xmax,ymax=walk.bounds
        gx,gy=np.meshgrid(np.arange(math.floor(xmin/STEP),math.ceil(xmax/STEP)+1),np.arange(math.floor(ymin/STEP),math.ceil(ymax/STEP)+1))
        xx,yy=(gx+.5)*STEP,(gy+.5)*STEP
        mask=contains_xy(walk,xx,yy)
        heights=np.full(mask.shape,np.nan)
        for area,floor in records:
            heights[intersects_xy(area,xx,yy)]=floor
        self.free=set(zip(gx[mask].astype(int).tolist(),gy[mask].astype(int).tolist()))
        self.floors={key:float(h) for key,h in zip(zip(gx[mask].tolist(),gy[mask].tolist()),heights[mask])}

    @staticmethod
    def point(key):
        return ((key[0]+.5)*STEP,(key[1]+.5)*STEP)

    def nearest(self,xy):
        key=(math.floor(xy[0]/STEP),math.floor(xy[1]/STEP))
        if key in self.free:return key
        nearby=[(x,y) for x in range(key[0]-4,key[0]+5) for y in range(key[1]-4,key[1]+5) if (x,y) in self.free]
        return min(nearby,key=lambda p:math.dist(xy,self.point(p))) if nearby else None

    def observe(self,s,tick):
        xy=(s['x'],s['y'])
        region=(math.floor(xy[0]/REGION),math.floor(xy[1]/REGION))
        if region!=self.last_region:
            self.regions[region]+=1
            self.last_region=region
        here=self.nearest(xy)
        if here not in self.entered:
            self.entered.add(here)
            self.last_new_tick=tick
        self.last_cell=here

    def neighbors(self,node):
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            nxt=(node[0]+dx,node[1]+dy)
            if nxt not in self.free or (node,nxt) in self.blocked:continue
            if dx and dy and ((node[0]+dx,node[1]) not in self.free or (node[0],node[1]+dy) not in self.free):continue
            diff=self.floors[nxt]-self.floors[node]
            if not -64<=diff<=24:continue
            yield nxt,math.hypot(dx,dy)*STEP

    def reject(self,tick):
        if self.path and self.last_cell is not None:
            nxt=self.path[0]
            # Блокируем конкретный неудачный переход, а не целую ветку маршрута.
            self.blocked.add((self.last_cell,nxt))
            self.blocked.add((nxt,self.last_cell))
            if len(self.path)>1:
                self.blocked.add((nxt,self.path[1]))
        self.path=[]
        self.objective=None

    def plan(self,s,tick,specific_xy=None):
        start=self.nearest((s['x'],s['y']))
        if start is None:
            self.exhausted=True
            return
        specific=self.nearest(specific_xy) if specific_xy is not None else None
        if specific_xy is not None and specific is None:
            self.path=[]
            self.exhausted=True
            self.goal_kind='unreachable'
            return
        queue=[(0,start)]
        costs,parent={start:0},{}
        goal=None
        fallback=None
        while queue:
            cost,node=heapq.heappop(queue)
            if cost!=costs[node]:continue
            p=self.point(node)
            region=(math.floor(p[0]/REGION),math.floor(p[1]/REGION))
            if node==specific:
                goal=node
                self.goal_kind='pickup'
                break
            if region not in self.regions and math.dist(p,(s['x'],s['y']))>=128:
                if specific is None:
                    goal=node
                    self.goal_kind='new_region'
                    break
                if fallback is None:fallback=node
            for nxt,edge in self.neighbors(node):
                nr=(math.floor(self.point(nxt)[0]/REGION),math.floor(self.point(nxt)[1]/REGION))
                proposed=cost+edge+self.regions[nr]*2
                if proposed<costs.get(nxt,float('inf')):
                    costs[nxt],parent[nxt]=proposed,node
                    heapq.heappush(queue,(proposed,nxt))
        if goal is None and specific is None:goal=fallback
        self.path=[]
        if goal is None:
            self.exhausted=True
            self.goal_kind='unreachable' if specific_xy is not None else 'exhausted'
            return
        self.exhausted=False
        self.objective=goal
        while goal!=start:
            self.path.append(goal)
            goal=parent[goal]
        self.path.reverse()

    def clear_segment(self,a,b):
        if self.walk is None or not self.walk.covers(LineString([a,b])):return False
        previous=self.nearest(a)
        for i in range(1,max(2,math.ceil(math.dist(a,b)/8))+1):
            count=max(2,math.ceil(math.dist(a,b)/8))
            point=(a[0]+(b[0]-a[0])*i/count,a[1]+(b[1]-a[1])*i/count)
            node=self.nearest(point)
            if node is None or previous is None:return False
            if not -64<=self.floors[node]-self.floors[previous]<=24:return False
            if (previous,node) in self.blocked:return False
            previous=node
        return True

    def steer(self,s,tick,objective=None):
        request_key=('point',self.nearest(objective)) if objective is not None else ('explore',)
        if request_key!=self.requested:
            self.path=[]
            self.requested=request_key
        xy=(s['x'],s['y'])
        # Бой/телепорт может увести далеко от старого пути. Строим путь от новой позиции.
        if self.path and min(math.dist(xy,self.point(p)) for p in self.path[:8])>96:
            self.path=[]
        while self.path and math.dist(self.point(self.path[0]),xy)<22:
            self.path.pop(0)
        if not self.path:
            request=(self.nearest(xy),request_key)
            if request==self.failed_request and tick<self.retry_tick:
                return [0,0,0,0,0,0,0],['navigation_exhausted']
            self.plan(s,tick,objective)
            if not self.path:
                self.failed_request,self.retry_tick=request,tick+35
            else:self.failed_request=None
        if not self.path:return [0,0,0,0,0,0,0],['navigation_exhausted']
        # Смотрим вперёд вдоль свободного коридора, а не целимся в каждый узел сетки.
        look=0
        for i,node in enumerate(self.path[:12]):
            point=self.point(node)
            if math.dist(point,xy)>96:break
            if self.clear_segment(xy,point):look=i
            else:break
        if look:self.path=self.path[look:]
        target=self.point(self.path[0])
        self.steering_target=target
        angle=math.degrees(math.atan2(target[1]-xy[1],target[0]-xy[0]))
        turn=-((angle-s['angle']+180)%360-180)
        if abs(turn)<1:turn=0
        return [float(abs(turn)<25),0,0,0,max(-6,min(6,turn*.7)),0,0],['waypoint']

    def state(self,tick):
        return {'known_cells':len(self.free),'visited_cells':len(self.entered),'regions':len(self.regions),
                'goal_kind':self.goal_kind,'exhausted':self.exhausted,'path_length':len(self.path),
                'no_new_cell_seconds':round((tick-self.last_new_tick)/35,2),
                'target':list(self.point(self.path[0])) if self.path else None}
