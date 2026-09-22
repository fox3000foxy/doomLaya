"""Supervised Doom adaptation: decision head, optionally final encoder layers."""
import os
os.environ['USE_TF']='0';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['USE_TORCH']='1'
import argparse,collections,hashlib,json,random,shutil,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from laya_runtime import base_checkpoint,choose_device,BASE_REPO,BASE_REVISION,LAYA_SOURCE_COMMIT

p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=4);p.add_argument('--output',default='checkpoints/laya-doom-head-v1');p.add_argument('--base');p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--encoder-last',type=int,default=0);p.add_argument('--data',default='training');p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
if a.epochs<1 or a.lr<=0 or not 0<=a.encoder_last<=28:p.error('epochs/lr must be positive; encoder-last must be in 0..28')
for name in ('train.json','validation.json'):
 rows=json.loads((Path(a.data)/name).read_text())
 if not rows:raise ValueError(f'Empty dataset: {name}')
 for row in rows:
  if row['kind'] not in ('command','weapon') or row['label'] not in row['question']['criteria']:raise ValueError(f'Invalid label in {name}')
 print(name,len(rows),'rows',hashlib.sha256((Path(a.data)/name).read_bytes()).hexdigest(),flush=True)
if a.validate_only:raise SystemExit(0)
if Path(a.output).exists():p.error('output already exists; choose a new directory')
import torch
from safetensors.torch import save_file
from laya import Agent
from laya.common import build_sequence
device=choose_device(a.device)
random.seed(771);torch.manual_seed(771);torch.set_num_threads(4)
base=base_checkpoint(a.base)
agent=Agent(str(base),device=device);model=agent.model
for name,param in model.named_parameters():param.requires_grad_(name.startswith(('head.','scorer.','type_emb.')))
if a.encoder_last:
 for layer in model.encoder.layers[-a.encoder_last:]:
  for param in layer.parameters():param.requires_grad_(True)
optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')],'lr':a.lr},
                            {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')],'lr':1e-5}],weight_decay=.01)

def prepare(path):
 result=[]
 for row in json.loads(Path(path).read_text()):
  q=row['question'];keys=list(q['criteria']);ids,markers=build_sequence(agent.tok,row['state'],{'t':'choice','ins':q['instructions'],'crit':q['criteria']},1024,256)
  result.append(dict(row,ids=ids,markers=markers,target=keys.index(row['label'])))
 return result
data=Path(a.data);train=prepare(data/'train.json');valid=prepare(data/'validation.json')

def batch(rows):
 n=len(rows);length=max(len(r['ids']) for r in rows);k=max(len(r['markers']) for r in rows)
 ids=torch.full((n,length),agent.tok.pad_token_id,dtype=torch.long);att=torch.zeros((n,length),dtype=torch.long)
 markers=torch.zeros((n,k),dtype=torch.long);mask=torch.zeros((n,k),dtype=torch.bool)
 for i,r in enumerate(rows):
  ids[i,:len(r['ids'])]=torch.tensor(r['ids']);att[i,:len(r['ids'])]=1
  markers[i,:len(r['markers'])]=torch.tensor(r['markers']);mask[i,:len(r['markers'])]=True
 return [t.to(device) for t in (ids,att,markers,mask,torch.zeros(n,dtype=torch.long))],torch.tensor([r['target'] for r in rows],device=device)

def evaluate():
 model.eval();correct=collections.Counter();total=collections.Counter()
 with torch.no_grad():
  for i in range(0,len(valid),4):
   chunk=valid[i:i+4];b,y=batch(chunk);z,_=model(*b)
   for row,guess in zip(chunk,z.argmax(-1).cpu().tolist()):
    total[row['kind']]+=1;correct[row['kind']]+=guess==row['target']
 return {k:correct[k]/total[k] for k in total}

out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
for directory in ('encoder','tokenizer'):shutil.copytree(base/directory,out/directory)
cfg=dict(agent.cfg,temperature=[1.,1.,1.],temperature_by_options={})
cfg['doom_adaptation']={'frozen_encoder':not bool(a.encoder_last),'trainable_encoder_layers':a.encoder_last,'training':str(data/'train.json'),'validation':str(data/'validation.json'),'data_sha256':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ('train.json','validation.json')},'seed':771,'base_checkpoint':base.name if a.base else f'{BASE_REPO}@{BASE_REVISION}/typed-decisions','learning_rate':a.lr,'encoder_learning_rate':1e-5 if a.encoder_last else None,'device':device,'laya_source_commit':LAYA_SOURCE_COMMIT}
shutil.copy2(__file__,out/'training_script.py')
(out/'rl_agent_config.json').write_text(json.dumps(cfg,indent=2))
metrics=[];start=time.perf_counter();initial=evaluate();print('BASE',initial,flush=True);best=sum(initial.values())
# Save the base first so a non-improving trial cannot silently export worse weights.
save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
for epoch in range(a.epochs):
 random.shuffle(train);model.train();model.encoder.eval();loss_sum=0
 for i in range(0,len(train),4):
  b,y=batch(train[i:i+4]);optimizer.zero_grad(set_to_none=True);z,_=model(*b)
  loss=torch.nn.functional.cross_entropy(z,y);loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
  optimizer.step();loss_sum+=float(loss.detach())
  if i%80==0:print('TRAIN',epoch+1,i,len(train),round(loss_sum/(i/4+1),3),round(time.perf_counter()-start,1),'grad',round(float(grad_norm),4),flush=True)
 score=evaluate();metrics.append({'epoch':epoch+1,'validation':score,'seconds':time.perf_counter()-start});print('VALID',metrics[-1],flush=True)
 if sum(score.values())>best:
  best=sum(score.values());save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
  (out/'best-epoch.json').write_text(json.dumps(metrics[-1],indent=2))
 (out/'training-metrics.json').write_text(json.dumps({'base':initial,'epochs':metrics},indent=2))
print('CHECKPOINT',out.resolve(),flush=True)
