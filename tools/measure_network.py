"""RTT TCP и HTTP без инференса; не выдаём HTTP TTFB за чистое время модели."""
import json
from pathlib import Path
import statistics
import subprocess
import time
import requests


def measure():
    result={}
    for name,url in [('openrouter','https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints'),
                     ('laya','http://127.0.0.1:8000/health')]:
        cold=[]
        for _ in range(8):
            p=subprocess.run(['curl','--silent','--show-error','--noproxy','*','--max-time','15','-o','/dev/null',
                              '-w','%{json}',url],capture_output=True,text=True,check=True)
            d=json.loads(p.stdout)
            cold.append({k:d[k] for k in ('http_code','time_namelookup','time_connect','time_appconnect','time_starttransfer','time_total')})
        session=requests.Session();session.trust_env=False
        warm=[]
        for i in range(9):
            start=time.perf_counter();r=session.get(url,timeout=(5,15));r.raise_for_status()
            elapsed=(time.perf_counter()-start)*1000
            if i:warm.append(elapsed)
        session.close()
        result[name]={'url':url,'cold_samples':cold,'warm_http_no_inference_ms':warm,
                      'tcp_handshake_rtt_ms_p50':statistics.median((x['time_connect']-x['time_namelookup'])*1000 for x in cold),
                      'tls_handshake_ms_p50':statistics.median((x['time_appconnect']-x['time_connect'])*1000 for x in cold) if name=='openrouter' else 0,
                      'warm_http_no_inference_ms_p50':statistics.median(warm)}
    return result

if __name__=='__main__':
    import sys
    data=measure();Path(sys.argv[1]).write_text(json.dumps(data,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a.endswith('p50')} for k,v in data.items()},indent=2))
