import numpy as np, json
from PIL import Image, ImageFilter
from scipy import ndimage
src=Image.open('bg.webp').convert('RGB'); im=np.array(src).astype(int)
r,g,b=im[...,0],im[...,1],im[...,2]
floor=(r>180)&(r-g>=14)&(r-g<=80)&(g-b>=-4)&(g-b<=52)&(r-b<=112)
# occluders: name, bbox, baseline  (chairs listed separately)
OCC=[
 ('desk1',[325,576,600,815],806),('chair1',[324,684,490,852],846),
 ('desk2',[703,464,950,695],684),('chair2',[706,564,836,702],699),
 ('desk3',[728,641,1024,890],872),('chair3',[736,761,905,940],936),
 ('desk4',[1016,514,1277,750],742),('chair4',[1024,621,1174,780],775),
 ('desk5',[1324,591,1604,847],838),('chair5',[1334,706,1494,880],875),
('leafBL',[0,795,205,941],99999),('leafBR',[1585,845,1670,941],99999),
]
EXCL={'desk1':[[325,576,418,628],[525,576,600,628]],'desk2':[[898,464,950,522]],'desk3':[[728,641,824,705],[940,641,1024,698]],'desk4':[[1016,514,1098,568],[1208,514,1277,566]],'desk5':[[1440,591,1604,603]]}
chairs={'desk%d'%i:'chair%d'%i for i in range(1,6)}
boxes={n:b for n,b,_ in OCC}
meta=[]
vis=np.array(src).copy()
for n,(x0,y0,x1,y1),base in OCC:
    m=~floor[y0:y1,x0:x1]
    if n.startswith('leaf'):
        gg=im[y0:y1,x0:x1]; m=(gg[...,1]>gg[...,0]-10)&(gg[...,1]>120)  # green leaves
        m=ndimage.binary_dilation(m,iterations=2)
    m=ndimage.binary_opening(m,iterations=1)
    if not n.startswith('leaf'): m=ndimage.binary_closing(m,iterations=4,border_value=0)
    lab,k=ndimage.label(m); 
    if k:
        sz=ndimage.sum(m,lab,range(1,k+1)); m=np.isin(lab,[i+1 for i,s in enumerate(sz) if s>=40])
    m=ndimage.binary_dilation(m,iterations=1)
    for ex0,ey0,ex1,ey1 in EXCL.get(n,[]): m[ey0-y0:ey1-y0, ex0-x0:ex1-x0]=False
    if n in chairs:  # remove chair pixels from desk
        cx0,cy0,cx1,cy1=boxes[chairs[n]]
        # keep desk pixels in chair box only above chair backrest top+12
        sub=m[max(0,cy0-y0+14):cy1-y0, max(0,cx0-x0):cx1-x0]; sub[:]=False
    a=Image.fromarray((m*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    crop=src.crop((x0,y0,x1,y1)).convert('RGBA'); crop.putalpha(a); crop.save(f'occ_{n}.png')
    vis[y0:y1,x0:x1][m]=(vis[y0:y1,x0:x1][m]*0.4+np.array([0,90,255])*0.6).astype(np.uint8)
    meta.append({'n':n,'x':x0,'y':y0,'base':base})
Image.fromarray(vis).save('occ_vis.png')
json.dump(meta,open('occ.json','w'))
