import numpy as np, json
from PIL import Image
from scipy import ndimage
D='/home/user/classroom/flat/assets'
foot=json.load(open('foot.json')); lj={o['n']:o for o in json.load(open('layers.json'))}
tiles=[]
for i in range(1,6):
    n=f'chair{i}'; o=lj[n]; A=np.array(Image.open(f'L_{n}.png').convert('RGBA')).astype(int)
    r,g,b,a=[A[...,k] for k in range(4)]; h,w=a.shape; yy,xx=np.mgrid[0:h,0:w]
    al=a>20
    orange=al&(r>215)&(g<200)&(g>130)&(b<135)
    seat=al&(r>232)&(g>=200)&(b<180)
    grey=al&(np.abs(r-b)<45)&(np.maximum(np.maximum(r,g),b)>95)
    FL,FR=foot[n][0],foot[n][1]; xl,xr=FL[0]-o['x'],FR[0]-o['x']
    # each backrest tube leans in a little toward the top
    band=np.zeros_like(al)
    for x0 in (xl,xr):
        cx=x0-(h-1-yy)*0.06
        band|=np.abs(xx-cx)<13
    # panel = largest orange blob
    lab,k=ndimage.label(orange); sz=ndimage.sum(orange,lab,range(1,k+1)); panel=lab==(np.argmax(sz)+1)
    ptop=np.where(panel.any(1))[0].min(); pbot=np.where(panel.any(1))[0].max()
    top=grey&(yy<ptop+8)
    back=panel|(grey&band)|top
    back=ndimage.binary_dilation(back,iterations=3)&al&~(seat&(yy>pbot-6))
    back|=ndimage.binary_fill_holes(back)&al&~seat
    B=A.copy(); B[...,3]=np.where(back,a,0); S=A.copy(); S[...,3]=np.where(back,0,a)
    Image.fromarray(B.astype(np.uint8)).save(f'{D}/back_{n}.webp',lossless=True)
    Image.fromarray(S.astype(np.uint8)).save(f'{D}/seat_{n}.webp',lossless=True)
    pv=Image.new('RGBA',(w*2+10,h),(40,60,90,255)); pv.alpha_composite(Image.fromarray(B.astype(np.uint8)),(0,0)); pv.alpha_composite(Image.fromarray(S.astype(np.uint8)),(w+10,0)); tiles.append(pv)
W=sum(t.width for t in tiles)+10*5; H=max(t.height for t in tiles)
out=Image.new('RGBA',(W,H),(0,0,0,255)); x=0
for t in tiles: out.alpha_composite(t,(x,0)); x+=t.width+10
out.resize((out.width*3//2,out.height*3//2)).save('backrest.png')
