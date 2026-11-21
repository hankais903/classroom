import numpy as np, json
from PIL import Image
from scipy import ndimage
old={o['n']:o for o in json.load(open('occ.json'))}
out=[]
for layer,prefix in [('desks.png','desk'),('chairs.png','chair')]:
    im=Image.open(layer).convert('RGBA'); A=np.array(im)
    m=A[...,3]>8
    lab,n=ndimage.label(ndimage.binary_dilation(m,iterations=8))
    sl=ndimage.find_objects(lab)
    comps=[]
    for i,s in enumerate(sl):
        cnt=(m[s]&(lab[s]==i+1)).sum()
        if cnt<500: continue
        comps.append((s,i+1,cnt))
    print(layer,'components',len(comps))
    for s,i,cnt in comps:
        y0,y1,x0,x1=s[0].start,s[0].stop,s[1].start,s[1].stop
        cx=(x0+x1)/2; cy=(y0+y1)/2
        # match to old cut-out of the same kind by nearest centre
        best=min([o for k,o in old.items() if k.startswith(prefix)], key=lambda o:(o['x']+150-cx)**2+(o['y']+120-cy)**2)
        crop=A[y0:y1,x0:x1].copy(); crop[...,3]=np.where(lab[y0:y1,x0:x1]==i,crop[...,3],0)
        Image.fromarray(crop).save(f"L_{best['n']}.png")
        out.append({'n':best['n'],'x':int(x0),'y':int(y0),'base':best['base'],'w':int(x1-x0),'h':int(y1-y0)})
        print(best['n'],x0,y0,x1,y1)
json.dump(out,open('layers.json','w'))
# preview on dark
pv=Image.new('RGBA',(1670,941),(40,60,90,255))
pv.alpha_composite(Image.open('desks.png').convert('RGBA')); pv.alpha_composite(Image.open('chairs.png').convert('RGBA'))
pv.crop((300,440,1620,941)).save('layers_dark.png')
