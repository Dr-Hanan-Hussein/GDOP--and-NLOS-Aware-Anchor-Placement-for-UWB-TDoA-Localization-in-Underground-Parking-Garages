"""make_fig1_architecture.py - Figure 1: schematic of the UWB-TDoA / V2I architecture (side view). Pure drawing, no data.
Usage: python make_fig1_architecture.py   ->  figures/fig1_architecture.png
"""
import os
os.makedirs("figures", exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Polygon
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":6.6,"mathtext.default":"regular"})
BLUE="#0b6fb3"; CAR="#e8c99b"; CARE="#b98a4d"; COL="#4d4d4d"; GRN="#2a8a57"; RED="#c0392b"; GREY="#6e6e6e"; LB="#e6f0f8"
W,H=6.3,3.05
fig=plt.figure(figsize=(W,H),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
FL=0.55; VS=0.5; HS=0.28            # floor y, vertical in/m, horizontal in/m (schematic, vertical exaggerated)
zy=lambda z: FL+z*VS
def car(x0,x1,h,fc,ec,ls="-",lw=0.7,wheels=True,z=3):
    ax.add_patch(FancyBboxPatch((x0,FL+0.07),x1-x0,zy(h)-FL-0.07,boxstyle="round,pad=0,rounding_size=0.06",fc=fc,ec=ec,lw=lw,ls=ls,zorder=z))
    ax.add_patch(Polygon([[x0+0.22*(x1-x0),zy(h)-0.02],[x0+0.32*(x1-x0),zy(h)-0.02+0.0]],closed=False,ec=ec,lw=0))
    if wheels:
        for wx in (x0+0.2*(x1-x0),x1-0.2*(x1-x0)): ax.add_patch(Circle((wx,FL+0.07),0.07,fc="#555",ec="white",lw=0.4,zorder=z+1))
# ---- garage (side view)
GX0,GX1=0.15,4.35
ax.plot([GX0,GX1],[FL,FL],color="k",lw=1.0,zorder=2)
ax.add_patch(Rectangle((GX0,zy(2.6)),GX1-GX0,0.10,fc="#d9d9d9",ec="none",hatch="////",zorder=1))
ax.text(GX0,2.82,"UWB-TDoA localization system for autonomous valet parking (schematic side view)",fontsize=7.1,fontweight="bold",va="bottom")
# parked car
car(2.95,4.25,1.5,CAR,CARE)
ax.text(3.6,zy(0.75),"parked car",fontsize=6,ha="center",va="center",zorder=5)
# ego vehicle
car(0.35,1.65,1.5,"white","k",lw=0.9)
ax.text(0.98,zy(0.95),"AVP vehicle",fontsize=6.2,ha="center",va="center",zorder=5)
tag=(1.65,zy(0.5)); ax.plot(*tag,"o",ms=4.2,mfc="k",mec="white",mew=0.5,zorder=9)
ax.annotate("on-board tag\n($z_{tag}$ = 0.5 m)",xy=tag,xytext=(1.2,zy(0.5)-0.46+0.55),fontsize=5.8,ha="center",va="top",arrowprops=dict(arrowstyle="-",lw=0.4),zorder=9) if False else None
# anchors
axs=[1.95,2.4,2.85,3.4,4.1]; zAN=2.0
for i,x in enumerate(axs):
    ax.plot([x,x],[zy(zAN),zy(2.6)],color=BLUE,lw=0.7,zorder=2)
    ax.plot(x,zy(zAN),"^",ms=6.5,mfc=BLUE,mec="white",mew=0.4,zorder=8)
    ax.text(x,zy(2.6)+0.17,(f"A{i}" if i else "A0 (ref.)"),fontsize=6.2,ha="center",va="bottom",color=BLUE,fontweight="bold")
# rays
def ray(xa,los,stop=None):
    p=np.array(tag); a=np.array([xa,zy(zAN)])
    if los: ax.plot([p[0],a[0]],[p[1],a[1]],color=GRN,lw=1.0,zorder=6)
    else:
        f=(stop-p[0])/(a[0]-p[0]); b=p+f*(a-p)
        ax.plot([p[0],b[0]],[p[1],b[1]],color=RED,lw=0.9,ls=(0,(3,2)),zorder=6)
        ax.plot([b[0],a[0]],[b[1],a[1]],color=RED,lw=0.5,ls=(0,(1,2)),alpha=0.55,zorder=6)
        ax.plot(*b,"x",ms=4,color=RED,mew=1.0,zorder=7)
for x in axs[:4]: ray(x,True)
ray(axs[4],False,stop=2.95)
# sync links (wireless clock sync)
for x0,x1 in zip(axs[:-1],axs[1:]):
    ax.add_patch(FancyArrowPatch((x0+0.05,zy(2.6)-0.045),(x1-0.05,zy(2.6)-0.045),connectionstyle="arc3,rad=0",arrowstyle="<|-|>",mutation_scale=4,lw=0.7,color="#4a8fc4",ls=(0,(1.2,1.2)),zorder=3))
# bracket "what this paper optimizes"
ax.plot([axs[0]-0.12,axs[0]-0.12,axs[-1]+0.12,axs[-1]+0.12],[zy(2.6)+0.62,zy(2.6)+0.68,zy(2.6)+0.68,zy(2.6)+0.62],color=BLUE,lw=0.7)
ax.text((axs[0]+axs[-1])/2,zy(2.6)+0.72,"anchor positions: the design variable of this paper",fontsize=6,ha="center",va="bottom",color=BLUE,style="italic")
# step 1 label + 'blink'
ax.text(0.35,zy(2.15),"(1) UWB blink:\nthe tag transmits,\nall anchors in LOS\nrecord its arrival time",fontsize=5.9,ha="left",va="center",linespacing=1.25,color="#1f1f1f")
ax.text(3.0,zy(2.15)+0.0,"",fontsize=1)
# ---- backhaul / engine / controller
EX0,EX1=5.08,6.27
def box(x0,y0,x1,y1,title,sub,fc):
    ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0,rounding_size=0.06",fc=fc,ec="#777",lw=0.7,zorder=3))
    ax.text((x0+x1)/2,y1-0.12,title,fontsize=6.9,fontweight="bold",ha="center",va="top",zorder=4)
    ax.text((x0+x1)/2,y1-0.36,sub,fontsize=5.9,ha="center",va="top",linespacing=1.3,zorder=4)
box(EX0,1.75,EX1,2.75,"Positioning engine","TDoA vs. reference A0:\n$D_{i0}$ = $c\\,\\Delta t_{i0}$ → solver → $\\hat{p}$\n(garage side)",LB)
box(EX0,0.62,EX1,1.37,"AVP controller","stall assignment,\npath planning",  "#e8f3ec")
# backhaul arrow anchors -> engine
yb=zy(2.6)+0.10+0.0
ax.plot([axs[0],axs[0]],[zy(2.6)+0.10,zy(2.6)+0.10],lw=0)
ax.add_patch(FancyArrowPatch((GX1-0.02,zy(2.6)+0.02),(EX0,2.25),connectionstyle="angle,angleA=0,angleB=90,rad=0",arrowstyle="-|>",mutation_scale=6,lw=1.0,color=GREY,zorder=4))
ax.text(4.38,zy(2.6)-0.04,"(2) timestamps\n(backhaul)",fontsize=5.9,ha="left",va="top",color=GREY,linespacing=1.2)
# engine -> controller
ax.add_patch(FancyArrowPatch(((EX0+EX1)/2,1.75),((EX0+EX1)/2,1.37),arrowstyle="-|>",mutation_scale=6,lw=1.0,color=GREY,zorder=4))
ax.text((EX0+EX1)/2+0.07,1.56,"(3) $\\hat{p}$",fontsize=5.9,ha="left",va="center",color=GREY)
# controller -> vehicle (V2I)
yv=0.2
ax.plot([(EX0+EX1)/2,(EX0+EX1)/2],[0.62,yv],color=GREY,lw=1.0,zorder=4)
ax.plot([(EX0+EX1)/2,0.98],[yv,yv],color=GREY,lw=1.0,zorder=4)
ax.add_patch(FancyArrowPatch((0.98,yv),(0.98,FL-0.04+0.0),arrowstyle="-|>",mutation_scale=6,lw=1.0,color=GREY,zorder=4,shrinkA=0,shrinkB=0))
ax.text(2.7,yv+0.05,"(4) V2I data link: position and driving commands to the vehicle",fontsize=5.9,ha="center",va="bottom",color=GREY)
# legend
from matplotlib.lines import Line2D
h=[Line2D([],[],color=GRN,lw=1.1),Line2D([],[],color=RED,lw=1,ls=(0,(3,2))),Line2D([],[],color="#4a8fc4",lw=0.9,ls=(0,(1.2,1.2)))]
ax.legend(h,["LOS range","NLOS (blocked by car)","wireless clock synchronization"],loc="upper left",bbox_to_anchor=(0.0,0.83),ncol=1,frameon=False,fontsize=5.8,handlelength=1.8,labelspacing=0.35,borderaxespad=0.1)
ax.text(4.3,0.0,"",fontsize=1)
fig.savefig("figures/fig1_architecture.png",dpi=300,facecolor="white")
