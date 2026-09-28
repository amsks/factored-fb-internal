#!/usr/bin/env python3
"""Draw the objective-selection counterexample for the deterministic path MDP."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/didactic2-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
INK, MUTED, RED, GREEN, BLUE, PALE = "#233342", "#536371", "#C13B44", "#148263", "#356AA5", "#EAF2FA"

def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=240, facecolor="white", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)

def mdp(ax):
    ax.set_xlim(-.9, 9.9); ax.set_ylim(-1.18, 1.0); ax.axis("off")
    xs = np.arange(4) * 3
    # The last two states are the region unlocked after crossing the gateway.
    ax.add_patch(plt.Rectangle((5.95, -.7), 3.95, 1.34, fc="#F4F8FC", ec="#C9D8E6", lw=1.2, zorder=0))
    ax.text(7.9, .61, "downstream region unlocked by the gateway", ha="center", fontsize=9, color=BLUE)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i]+.48,.25),(xs[i+1]-.48,.25),arrowstyle="-|>",mutation_scale=13,lw=1.6,color=INK))
        ax.add_patch(FancyArrowPatch((xs[i+1]-.48,-.25),(xs[i]+.48,-.25),arrowstyle="-|>",mutation_scale=13,lw=1.6,color=INK))
        ax.text((xs[i]+xs[i+1])/2,.42,"R",ha="center",fontsize=10,color=INK)
        ax.text((xs[i]+xs[i+1])/2,-.47,"L",ha="center",fontsize=10,color=INK)
    for i,x in enumerate(xs):
        goal=i==1
        gateway = i == 1
        ax.add_patch(Circle((x,0),.43,fc="#E3F3EB" if gateway else "white",ec=GREEN if gateway else INK,lw=2.4 if gateway else 2))
        ax.text(x,0,f"$s_{i}$",ha="center",va="center",fontsize=18,color=GREEN if goal else INK)
    ax.text(xs[0],-.72,"start / outside",ha="center",fontsize=10,color=INK)
    ax.text(xs[1],.61,"gateway",ha="center",fontsize=9.5,color=GREEN,weight="bold")
    ax.text(xs[1],-.72,"reward on arrival",ha="center",fontsize=9.5,color=GREEN)
    ax.text(4.5,.86,"Bottleneck: every downstream state requires crossing $s_1$",ha="center",fontsize=12.5,weight="bold",color=GREEN)
    ax.text(4.5,-1.08,"deterministic dynamics; continuing goal; $\\gamma=0.8$",ha="center",fontsize=10,color=MUTED)

def representation(ax):
    ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    xs=np.linspace(.12,.88,4)
    for x,label in zip(xs,["$s_0$","$s_1$","$s_2$","$s_3$"]): ax.text(x,.92,label,ha="center",fontsize=12,color=INK)
    bad=np.array([[1,0],[255/257,32/257],[0,1],[-32/257,255/257]])
    good=np.array([[1,1],[1,1],[1,-1],[1,-1]])/np.sqrt(2)
    for y,data,title,color,subtitle in [(0.58,bad,"lower square loss",RED,"wrong start action"),(0.18,good,"feasible control solution",GREEN,"right start action")]:
        ax.plot(xs,data[:,0]*.10+y,color=BLUE,lw=2,marker="o",ms=5)
        ax.plot(xs,data[:,1]*.10+y,color="#9A6BB2",lw=2,marker="o",ms=5)
        ax.text(.02,y+.055,title,ha="left",fontsize=11,weight="bold",color=color)
        ax.text(.98,y-.03,subtitle,ha="right",fontsize=10,color=color)
    ax.text(.5,.03,"Both have dimension 2, row norm $\\sqrt{2}$, and $B^\\top B/4=I$",ha="center",fontsize=10,color=MUTED)

def objective(ax):
    names=["correct\ncontrol","square-loss\nselected"]
    vals=[.1699654,.0806398]
    bars=ax.bar(names,vals,color=[GREEN,RED],width=.56)
    ax.set_ylim(0,.20); ax.set_ylabel("global bilinear square objective",fontsize=10)
    ax.set_title("The objective prefers the wrong representation",fontsize=12,weight="bold",color=INK)
    ax.grid(axis="y",alpha=.2); ax.spines[['top','right']].set_visible(False)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.006,f"{v:.5f}",ha="center",fontsize=10)

def categorical(ax):
    # Explicit feasible rank-two-logit candidate from the exact certificate.
    # Row 0 is the left/self successor at s0; row 1 is the right/goal successor.
    left = np.array([0.50067463, 0.27477606, 0.12346484, 0.10108446])
    right = np.array([0.27657795, 0.35870226, 0.24044531, 0.12427448])
    x = np.arange(4); width=.34
    ax.bar(x-width/2,left,width,label="left action",color="#8A9BA8")
    ax.bar(x+width/2,right,width,label="right action",color=BLUE)
    ax.axvspan(.55,1.45,color="#E3F3EB",zorder=0)
    ax.text(1,.96,"goal state",ha="center",va="top",fontsize=10,weight="bold",color=GREEN)
    ax.set_xticks(x,["$s_0$","$s_1$","$s_2$","$s_3$"])
    ax.set_ylim(0,.62); ax.set_ylabel("predicted successor probability",fontsize=10)
    ax.set_title("Categorical successor model preserves the goal probability",fontsize=12,weight="bold",color=INK)
    ax.legend(frameon=False,ncol=2,loc="upper right")
    ax.spines[['top','right']].set_visible(False); ax.grid(axis="y",alpha=.2)
    ax.text(1,-.22,"$q(s_1\\mid s_0,\\mathrm{right})=0.359 > q(s_1\\mid s_0,\\mathrm{left})=0.275$",transform=ax.transAxes,ha="center",fontsize=10,color=GREEN)

def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig=plt.figure(figsize=(13,7.8),facecolor="white")
    gs=fig.add_gridspec(3,2,height_ratios=[.85,1.05,1.0],wspace=.28,hspace=.78)
    ax=fig.add_subplot(gs[0,:]); mdp(ax)
    ax=fig.add_subplot(gs[1,0]); representation(ax)
    ax=fig.add_subplot(gs[1,1]); objective(ax)
    ax=fig.add_subplot(gs[2,:]); categorical(ax)
    fig.suptitle("Didactic-2: failure of the bilinear square-loss objective",fontsize=18,weight="bold",color=INK,y=.98)
    fig.text(.5,.018,"A correct normalized rank-2 solution exists, but the global square-loss optimum reverses the gateway action and blocks the downstream region.",ha="center",fontsize=10.5,color=INK)
    save(fig,"bilinear_squareloss_counterexample")

if __name__ == "__main__": main()
