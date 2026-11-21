import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
im=np.array(Image.open('cat.webp').convert('RGB')).astype(int)
CROPS={'front':(655,88,932,398),'side':(958,88,1200,398),'back':(1232,88,1505,398),
'happy':(512,492,708,652),'surprised':(712,486,905,652),'sad':(912,492,1098,652),'sleepface':(1322,492,1512,652),
'walk':(62,735,315,990),'sit':(342,755,570,990),'lie':(612,795,915,990),'jump':(1228,720,1505,1005),'wave':(115,12,540,525)}
out={}
for k,(x0,y0,x1,y1) in CROPS.items():
    a=im[y0:y1,x0:x1]; r,g,b=a[...,0],a[...,1],a[...,2]
    light=(np.minimum(np.minimum(r,g),b)>185)&((r-b)<45)
    lab,n=ndimage.label(light)
    border=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
    bg=np.isin(lab,list(border))
    fg=~bg
    lab2,n2=ndimage.label(fg)
    sizes=ndimage.sum(fg,lab2,range(1,n2+1))
    keep=np.isin(lab2,[i+1 for i,s in enumerate(sizes) if s>0.15*sizes.max()])
    keep=ndimage.binary_fill_holes(keep) if k not in ('side',) else keep
    alpha=Image.fromarray((keep*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    rgba=Image.fromarray(a.astype(np.uint8)).convert('RGBA'); rgba.putalpha(alpha)
    bb=rgba.getbbox(); rgba=rgba.crop(bb); rgba.save(f'sp_{k}.png'); out[k]=rgba.size
print(out)
# preview sheet on dark bg
W=sum(s[0] for s in out.values())+20*len(out); H=max(s[1] for s in out.values())+20
pv=Image.new('RGBA',(W,H),(60,50,80,255)); x=10
for k in out:
    s=Image.open(f'sp_{k}.png'); pv.alpha_composite(s,(x,10)); x+=s.size[0]+20
pv.save('preview.png')
