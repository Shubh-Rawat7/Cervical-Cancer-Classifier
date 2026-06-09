import torch, pprint
import sys
p='Checkpoints/best_model.pt'
ck = torch.load(p, map_location='cpu')
if isinstance(ck, dict):
    sd = ck.get('state_dict', ck)
else:
    sd = ck
# strip prefixes
new = {}
for k,v in sd.items():
    nk = k
    for prefix in ('module.', 'ema_model.', 'model.'):
        if nk.startswith(prefix):
            nk = nk[len(prefix):]
    new[nk]=v
sd=new
keys = sorted(k for k in sd.keys() if k.startswith('head.'))
print('Found head keys:', keys)
for k in keys:
    print(k, sd[k].shape)
meta = ck.get('config', {}) if isinstance(ck, dict) else {}
print('metadata keys:', list(meta.keys()))
print('metadata sample:', {k: meta[k] for k in ('embed_dim','num_classes','backbone') if k in meta})
if 'class_names' in meta:
    print('checkpoint class_names:', meta['class_names'])
