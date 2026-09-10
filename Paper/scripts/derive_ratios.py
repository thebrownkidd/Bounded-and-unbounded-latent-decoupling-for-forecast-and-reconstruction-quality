"""Every ratio/comparison the paper states in prose, computed from paper_numbers.json.

If a sentence in the paper says "X times" or "N% better", the number comes from
here and nowhere else.
"""
import json, io

SP = "C:/Users/ARPITG~1/AppData/Local/Temp/claude/c--Users-ArpitGoel-Documents-GitHub-Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality/cdd20b34-1bb7-4cf8-a41f-3ee2fc81363c/scratchpad/"
J = json.load(open(SP + "paper_numbers.json"))
R, D, P = J["results"], J["data"], J["params"]
N = D["n_obs"]
FLOOR = D["noise_floor_mse"]
OURS = R["Ours (three-phase)"]
d_of = {"Ours (three-phase)": 3, "PINN (true physics)": 3, "PINN (rho=26)": 3,
        "phi=0, latent=3": 3}
for k in R:
    d_of.setdefault(k, 16)

out = {}

# ---- forecast leadership
# Both "Ours" rows are the proposed model, so neither counts as a rival. The
# one-stage ablation in particular outscores every baseline, so leaving it in
# would silently make it "the best baseline".
OURS_KEYS = {"Ours (three-phase)", "Ours (one-stage)"}
trained = {k: v for k, v in R.items()
           if k not in ("persistence", "climatology", "phi=0, latent=3")}
rivals = {k: v for k, v in trained.items() if k not in OURS_KEYS}
best_rival = max(rivals.items(), key=lambda kv: kv[1]["vpt"])
out["best_rival_name"] = best_rival[0]
out["best_rival_vpt"] = best_rival[1]["vpt"]
out["ours_vpt"] = OURS["vpt"]
out["ours_vpt_over_best_rival"] = OURS["vpt"] / best_rival[1]["vpt"]
out["ours_vpt_over_persistence"] = OURS["vpt"] / R["persistence"]["vpt"]
out["ours_vpt_steps"] = OURS["vpt"] * D["steps_per_lyapunov"]
out["best_rival_vpt_steps"] = best_rival[1]["vpt"] * D["steps_per_lyapunov"]

# ---- long horizon
out["ours_nrmse5"] = OURS["fcst_nrmse_full"]
out["clim_nrmse5"] = R["climatology"]["fcst_nrmse_full"]
out["ours_nrmse5_over_clim"] = OURS["fcst_nrmse_full"] / R["climatology"]["fcst_nrmse_full"]
out["models_below_climatology"] = [k for k, v in R.items()
                                   if v["fcst_nrmse_full"] < R["climatology"]["fcst_nrmse_full"]]

# ---- reconstruction, absolute and capacity-normalised
rows = []
for k, v in R.items():
    if "recon_mse" not in v or v.get("recon_mse") is None:
        continue
    d = d_of[k]
    bound = (N - d) / N
    rows.append({"model": k, "d": d, "recon_mse": v["recon_mse"],
                 "over_floor": v["recon_over_floor"],
                 "bound": bound, "over_own_optimum": v["recon_over_floor"] / bound})
rows.sort(key=lambda r: r["over_own_optimum"])
out["recon_capacity_table"] = rows
out["ours_over_floor"] = OURS["recon_over_floor"]
out["ours_over_own_optimum"] = OURS["recon_over_floor"] / ((N - 3) / N)
sub = [r for r in rows if r["over_floor"] < 1.0]
out["n_models_below_floor"] = len(sub)
out["models_below_floor"] = [r["model"] for r in sub]
best_recon_rival = min((r for r in rows if r["model"] not in OURS_KEYS
                        and r["model"] != "phi=0, latent=3"),
                       key=lambda r: r["over_own_optimum"])
out["best_recon_rival"] = best_recon_rival

# ---- stability
out["ours_divergence"] = OURS["divergence"]
out["diverging_baselines"] = {k: v["divergence"] for k, v in rivals.items()}
out["n_rivals_full_divergence"] = sum(1 for v in rivals.values() if v["divergence"] >= 0.98)

# ---- passthrough vs prediction
pt = {}
for k, v in J["passthrough"].items():
    pred_proj = v["d"] / N
    pred_denoise = 3 / N
    pt[k] = {**v, "pred_rank_d_projection": pred_proj, "pred_pure_denoiser": pred_denoise,
             "effective_dims_copied": v["passthrough"] * N}
out["passthrough"] = pt

# ---- pca check of the capacity bound
out["pca_check"] = {d: {"measured": J["pca_rank_over_floor"][str(d)],
                        "predicted": (N - d) / N} for d in (3, 16, 24, 30)}

# ---- parameter matching
out["param_match"] = {
    "target": P["target"],
    "joint_pct_off": 100 * (P["joint_params"] - P["target"]) / P["target"],
    "sigmoid_pct_off": 100 * (P["sigmoid_params"] - P["target"]) / P["target"],
    "pinn_pct_off": 100 * (P["pinn_params"] - P["target"]) / P["target"],
    "max_abs_pct_off": max(abs(100 * (P[x] - P["target"]) / P["target"])
                           for x in ("joint_params", "sigmoid_params", "pinn_params")),
}

# ---- climate
out["climate"] = J["climate_50LT"]
cl = J["climate_50LT"]
out["ours_wasserstein"] = cl["Ours (three-phase)"]["wasserstein"]
out["best_climate_rival"] = min(((k, v["wasserstein"]) for k, v in cl.items()
                                 if k != "Ours (three-phase)"), key=lambda kv: kv[1])

# ---- lead-time profile
PR = J["profile"]
B, LEADS = PR["by_model"], PR["leads_LT"]
i1 = LEADS.index(1.0)
i50 = LEADS.index(50.0)
trained_p = [k for k in B if k not in ("persistence", "climatology", "phi=0, latent=3")]
riv_p = [k for k in trained_p if k not in OURS_KEYS]

out["acc_horizon_ours"] = B["Ours (three-phase)"]["acc_horizon_LT"]
best_ah = max(riv_p, key=lambda k: B[k]["acc_horizon_LT"])
out["acc_horizon_best_rival"] = {"model": best_ah, "value": B[best_ah]["acc_horizon_LT"]}
out["acc_horizon_ratio"] = (B["Ours (three-phase)"]["acc_horizon_LT"]
                            / B[best_ah]["acc_horizon_LT"])
out["acc_horizon_tied_at_best"] = [k for k in riv_p
                                   if abs(B[k]["acc_horizon_LT"]
                                          - B[best_ah]["acc_horizon_LT"]) < 1e-9]

out["mae1_ours"] = B["Ours (three-phase)"]["mae"][i1]
best_mae = min(riv_p, key=lambda k: B[k]["mae"][i1])
out["mae1_best_rival"] = {"model": best_mae, "value": B[best_mae]["mae"][i1]}
out["mae1_ratio"] = B[best_mae]["mae"][i1] / B["Ours (three-phase)"]["mae"][i1]

out["acc1_ours"] = B["Ours (three-phase)"]["acc"][i1]
out["ss1_ours"] = B["Ours (three-phase)"]["ss_persist"][i1]
out["varratio_ours_50"] = B["Ours (three-phase)"]["var_ratio"][i50]
out["varratio_phi01_50"] = B["phi=0.1"]["var_ratio"][i50]
# who keeps a sane amplitude all the way out, and who explodes
out["varratio_50_all"] = {k: B[k]["var_ratio"][i50] for k in B}
out["n_exploding_50"] = sum(1 for k in trained_p if B[k]["var_ratio"][i50] > 2.0)

io.open(SP + "paper_ratios.json", "w", encoding="utf-8").write(json.dumps(out, indent=1))
print(f"\nACC horizon: ours {out['acc_horizon_ours']:.3f} LT vs "
      f"{out['acc_horizon_best_rival']['value']:.3f} ({out['acc_horizon_best_rival']['model']}) "
      f"= {out['acc_horizon_ratio']:.2f}x   tied at best: {out['acc_horizon_tied_at_best']}")
print(f"MAE@1LT: ours {out['mae1_ours']:.3f} vs {out['mae1_best_rival']['value']:.3f} "
      f"({out['mae1_best_rival']['model']}) = {out['mae1_ratio']:.2f}x lower")
print(f"ACC@1LT ours {out['acc1_ours']:.3f} | skill vs persistence@1LT {out['ss1_ours']:.3f}")
print(f"VarRatio@50LT ours {out['varratio_ours_50']:.3f}, phi=0.1 {out['varratio_phi01_50']:.3f}, "
      f"exploding models {out['n_exploding_50']}")

print(f"ours VPT {out['ours_vpt']:.4f} LT ({out['ours_vpt_steps']:.0f} steps)")
print(f"best rival: {out['best_rival_name']} at {out['best_rival_vpt']:.4f} "
      f"-> ours is {out['ours_vpt_over_best_rival']:.3f}x")
print(f"ours NRMSE@5LT {out['ours_nrmse5']:.4f} vs climatology {out['clim_nrmse5']:.4f} "
      f"= {out['ours_nrmse5_over_clim']:.3f}x")
print(f"below climatology: {out['models_below_climatology']}")
print(f"ours recon {out['ours_over_floor']:.3f}x floor, {out['ours_over_own_optimum']:.3f}x own optimum")
print(f"models below floor: {out['n_models_below_floor']} -> {out['models_below_floor']}")
print(f"best recon rival by own-optimum: {best_recon_rival['model']} "
      f"{best_recon_rival['over_own_optimum']:.3f}x")
print(f"param match: max {out['param_match']['max_abs_pct_off']:.2f}% off {P['target']:,}")
print("pca check:", {k: (round(v['measured'],3), round(v['predicted'],3)) for k,v in out['pca_check'].items()})
print("passthrough:", {k: round(v['passthrough'],3) for k,v in pt.items()})
print(f"ours wasserstein {out['ours_wasserstein']:.4f}, best rival {out['best_climate_rival']}")
