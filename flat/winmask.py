import numpy as np, json
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
D='/home/user/classroom/flat/assets'
bg=Image.open('bg2c.png').convert('RGB'); A=np.array(bg).astype(int)
X0,Y0,X1,Y1=100,130,640,400
PANES=[
 [(141,141),(220,153),(220,208),(131,200)],
 [(238,157),(328,169),(328,218),(238,211)],
 [(126,226),(219,231),(219,377),(126,382)],
 [(238,234),(315,238),(315,368),(238,375)],
 [(329,240),(360,243),(360,362),(329,365)],
 [(100,226),(122,226),(122,382),(100,382)],
 [(503,200),(540,202),(540,238),(503,236)],
 [(551,206),(598,211),(598,244),(551,241)],
 [(466,252),(540,254),(540,352),(466,352)],
 [(551,256),(598,259),(598,289),(577,289),(577,352),(551,352)],
]
P=Image.new('L',(X1-X0,Y1-Y0),0); d=ImageDraw.Draw(P)
for q in PANES: d.polygon([(x-X0,y-Y0) for x,y in q],fill=255)
m=np.array(P)>0
# bunny lamp on the sill sits in front of the glass
E=Image.new('L',P.size,0); ImageDraw.Draw(E).ellipse([150-X0,350-Y0,218-X0,400-Y0],fill=255); m&=~(np.array(E)>0)
w=A[Y0:Y1,X0:X1]; r,g,b=w[...,0],w[...,1],w[...,2]; mx=np.maximum(np.maximum(r,g),b); mn=np.minimum(np.minimum(r,g),b)
curtain=(r>225)&(g>190)&(r-b>45)&(b<205)
yy,xx=np.mgrid[Y0:Y1,X0:X1]
leaf=(b<115)&(g>140)&(xx>470)&(yy>290)
outline=mx<110
inside=~(curtain|leaf|outline)
inside=ndimage.binary_opening(inside,iterations=1)
m&=inside
# fill pinholes inside each pane (tree highlights), but keep real foreground cut-outs
holes=ndimage.binary_fill_holes(m)&~m
hl,hk=ndimage.label(holes); hs=ndimage.sum(holes,hl,range(1,hk+1)); m|=np.isin(hl,[i+1 for i,v in enumerate(hs) if v<40])
m=ndimage.binary_erosion(m,iterations=1)
lab,k=ndimage.label(m); sz=ndimage.sum(m,lab,range(1,k+1)); m=np.isin(lab,[i+1 for i,s in enumerate(sz) if s>60])
mk=Image.fromarray((m*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
out=Image.new('RGBA',mk.size,(255,255,255,255)); out.putalpha(mk); out.save(f'{D}/win_mask.webp',lossless=True)
json.dump({'x':X0,'y':Y0},open(D+'/window.json','w'))
pv=bg.convert('RGBA').crop((X0,Y0,X1,Y1)); t=Image.new('RGBA',pv.size,(255,0,0,255)); t.putalpha(Image.fromarray((m*130).astype(np.uint8))); pv.alpha_composite(t); pv.resize((pv.width*2,pv.height*2)).save('win_vis.png')
