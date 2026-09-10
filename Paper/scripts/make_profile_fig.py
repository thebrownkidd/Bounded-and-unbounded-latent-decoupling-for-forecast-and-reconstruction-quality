import sys, json
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
from Utils import PALETTE, UseFigureStyle

SP = ("C:/Users/ARPITG~1/AppData/Local/Temp/claude/c--Users-ArpitGoel-Documents-GitHub-"
      "Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality/"
      "cdd20b34-1bb7-4cf8-a41f-3ee2fc81363c/scratchpad/")
J = json.load(open(SP + "paper_numbers.json"))
FIG = ROOT / "Paper" / "figs"
PR = J["profile"]
BY, LEADS = PR["by_model"], PR["leads_LT"]

SHOW = ["Ours (three-phase)", "phi=0.1", "phi=0.5", "AEGRU+sigmoid",
        "PINN (true physics)", "persistence"]
LBL = {"Ours (three-phase)": "Ours", "phi=0.1": r"$\phi$=0.1", "phi=0.5": r"$\phi$=0.5",
       "AEGRU+sigmoid": "AEGRU+sigmoid", "PINN (true physics)": "PINN",
       "persistence": "persistence"}
PANELS = [("mae", "MAE", True), ("acc", "ACC", False),
          ("ss_persist", "skill vs persist.", False),
          ("var_ratio", "variance ratio", True)]

UseFigureStyle()
fig, axes = plt.subplots(1, 4, figsize=(12.2, 2.5), dpi=300)
for ax, (key, ylab, logy) in zip(axes, PANELS):
    for j, name in enumerate(SHOW):
        col = PALETTE[0] if name == "Ours (three-phase)" else PALETTE[1 + (j % (len(PALETTE) - 1))]
        lw = 1.8 if name == "Ours (three-phase)" else 1.0
        ax.plot(LEADS, BY[name][key], marker="o", ms=2.6, lw=lw, color=col,
                label=LBL[name])
    ax.set_xscale("log")
    ax.set_xlabel("lead time (Lyapunov times)")
    ax.set_ylabel(ylab)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    if key == "acc":
        ax.axhline(PR["acc_threshold"], ls=":", lw=1.0, color="0.35")
        ax.text(0.55, PR["acc_threshold"] + 0.04, "0.6", fontsize=5.2, color="0.35")
        ax.set_ylim(-0.3, 1.05)
    if key == "ss_persist":
        ax.axhline(0.0, ls=":", lw=1.0, color="0.35")
        ax.set_ylim(-1.1, 1.05)
    if key == "var_ratio":
        ax.axhline(1.0, ls=":", lw=1.0, color="0.35")
    if logy:
        ax.set_yscale("log")
axes[0].legend(loc="upper center", bbox_to_anchor=(2.4, -0.34), ncol=6, fontsize=6.4)
fig.tight_layout()
fig.savefig(FIG / "paper_profile.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "paper_profile.png")
