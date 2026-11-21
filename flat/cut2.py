import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
im=np.array(Image.open('cat.webp').convert('RGB')).astype(int)
CROPS={'front':(655,88,932,398),'side':(958,88,1200,398),'back':(1232,88,1505,398),
'happy':(512,492,708,652),'surprised':(712,486,905,652),'sad':(912,492,1098,652),'sleepface':(1322,492,1512,652),
'walk':(62,735,315,990),'sit':(342,755,570,990),'lie':(612,795,915,990),'jump':(1228,720,1505,1005),'wave':(115,12,540,525)}
for k,(x0,y0,x1,y1) in CROPS.items():
    a=im[y0:y1,x0:x1]; r,g,b=a[...,0],a[...,1],a[...,2]
    mn=np.minimum(np.minimum(r,g),b); mx=np.maximum(np.maximum(r,g),b)
    loose=(mn>185)&((r-b)<45)
    lab,n=ndimage.label(loose)
    border=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
    bg=np.isin(lab,list(border))
    # enclosed paper-white or pink-shadow pockets (cat white is warm, paper is neutral)
    pocket=((mn>=251)&((mx-mn)<=3))
    pl,pn=ndimage.label(pocket)
    if pn:
        dark=mn<120; kept=[]
        for i in range(1,pn+1):
            m=pl==i
            if m.sum()<25: continue
            ring=ndimage.binary_dilation(m,iterations=3)&~m
            if dark[ring].mean()>=0.5: kept.append(i)
        bg|=np.isin(pl,kept)
        if kept: print(k,'pockets removed',len(kept))
    fg=~bg
    fg=ndimage.binary_opening(fg,iterations=1)
    l2,n2=ndimage.label(fg); sz=ndimage.sum(fg,l2,range(1,n2+1))
    fg=np.isin(l2,[i+1 for i,s in enumerate(sz) if s>0.15*sz.max()])
    fg=ndimage.binary_erosion(fg,iterations=1)
    alpha=Image.fromarray((fg*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    rgba=Image.fromarray(a.astype(np.uint8)).convert('RGBA'); rgba.putalpha(alpha)
    rgba=rgba.crop(rgba.getbbox()); rgba.save(f'sp_{k}.png')
ks=['front','side','back','sit','lie','jump','walk']
ims=[Image.open(f'sp_{k}.png') for k in ks]
W=sum(i.width for i in ims)+20*8; H=max(i.height for i in ims)+20
pv=Image.new('RGBA',(W,H),(232,170,160,255)); x=10
for i in ims: pv.alpha_composite(i,(x,10)); x+=i.width+20
pv.save('pv2.png')
