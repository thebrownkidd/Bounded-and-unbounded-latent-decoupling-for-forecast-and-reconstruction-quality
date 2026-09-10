"""Check that every headline number in the compiled .tex came from the JSON.

Not circular: it recomputes each quantity from paper_numbers.json / paper_ratios
.json independently of how write_paper.py formatted it, then asserts the
resulting string is actually present in the .tex. That catches a wrong
substitution key, a stale placeholder, or a hand-typed number.
"""
import json, re, io

SP = ("C:/Users/ARPITG~1/AppData/Local/Temp/claude/c--Users-ArpitGoel-Documents-GitHub-"
      "Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality/"
      "cdd20b34-1bb7-4cf8-a41f-3ee2fc81363c/scratchpad/")
J = json.load(open(SP + "paper_numbers.json"))
Rt = json.load(open(SP + "paper_ratios.json"))
tex = io.open("Paper/fmts2026.tex", encoding="utf-8").read()
R, D, P, CL = J["results"], J["data"], J["params"], J["climate_50LT"]
PR, BD, PT = J["profile"], J["boundedness"], J["passthrough"]
O = R["Ours (three-phase)"]
I1 = PR["leads_LT"].index(1.0)

checks = [
    ("ours VPT", f"{O['vpt']:.3f}"),
    ("ours VPT std", f"{O['vpt_std']:.3f}"),
    ("best rival VPT", f"{R['phi=0.1']['vpt']:.3f}"),
    ("VPT ratio", f"{Rt['ours_vpt_over_best_rival']:.2f}"),
    ("VPT steps", f"{Rt['ours_vpt_steps']:.0f}"),
    ("ours recon MSE", f"{O['recon_mse']:.5f}"),
    ("ours x floor", f"{O['recon_over_floor']:.3f}"),
    ("ours err 5LT", f"{O['fcst_nrmse_full']:.3f}"),
    ("climatology err 5LT", f"{R['climatology']['fcst_nrmse_full']:.3f}"),
    ("noise floor", f"{D['noise_floor_mse']:.6f}"),
    ("target params", f"{P['target']:,}"),
    ("param pct off", f"{Rt['param_match']['max_abs_pct_off']:.2f}"),
    ("steps per lyap", f"{D['steps_per_lyapunov']:.1f}"),
    ("pooled train", f"{D['pooled_train_steps']:,}"),
    ("pca16 measured", f"{Rt['pca_check']['16']['measured']:.3f}"),
    ("pca16 predicted", f"{Rt['pca_check']['16']['predicted']:.3f}"),
    ("passthrough phi=0", f"{PT['phi=0']['passthrough']:.3f}"),
    ("passthrough ours", f"{PT['Ours (three-phase)']['passthrough']:.3f}"),
    ("passthrough phi=0.5", f"{PT['phi=0.5']['passthrough']:.3f}"),
    ("passthrough phi=1", f"{PT['phi=1']['passthrough']:.3f}"),
    ("own optimum ours", f"{Rt['ours_over_own_optimum']:.3f}"),
    ("own optimum rival", f"{Rt['best_recon_rival']['over_own_optimum']:.3f}"),
    ("wasserstein ours", f"{CL['Ours (three-phase)']['wasserstein']:.4f}"),
    ("wasserstein pinn", f"{CL['PINN (true physics)']['wasserstein']:.4f}"),
    ("b absmax", f"{BD['b_absmax']:.1f}"),
    ("phi0 VPT", f"{R['phi=0']['vpt']:.3f}"),
    # lead-time profile
    ("MAE@1LT ours", f"{Rt['mae1_ours']:.3f}"),
    ("MAE@1LT rival", f"{Rt['mae1_best_rival']['value']:.3f}"),
    ("ACC horizon ours", f"{Rt['acc_horizon_ours']:.2f}"),
    ("ACC horizon rival", f"{Rt['acc_horizon_best_rival']['value']:.2f}"),
    ("ACC horizon ratio", f"{Rt['acc_horizon_ratio']:.2f}"),
    ("varratio ours 50LT", f"{Rt['varratio_ours_50']:.3f}"),
    ("varratio phi=0.1 50LT", f"{Rt['varratio_phi01_50']:.3f}"),
    ("n exploding", str(Rt["n_exploding_50"])),
]

bad = 0
for name, computed in checks:
    if computed not in tex:
        bad += 1
        print(f"  MISSING from tex: {name} = '{computed}'")
print(f"{len(checks) - bad}/{len(checks)} computed values found verbatim in the .tex")

print("leftover placeholders:", bool(re.search(r"__[A-Z0-9_]+__", tex)))
print("formfeed chars:", chr(12) in tex)
print("unicode em/en dash:", (chr(8212) in tex) or (chr(8211) in tex))
print("literal --- :", len(re.findall(r"(?<!-)---(?!-)", tex)))
semis = [m for m in re.finditer(r";", tex)]
gram = [m for m in semis if tex[max(0, m.start() - 1)] != "\\"]
print(f"semicolons total {len(semis)}, grammatical: {len(gram)}")
