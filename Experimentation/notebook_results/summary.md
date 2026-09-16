# Extracted results from LatentForecastComparisonMultiSeries.ipynb

## Cell 2: final output (1 stream records collapsed)
    C:\Users\ArpitGoel\AppData\Roaming\Python\Python313\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
    from .autonotebook import tqdm as notebook_tqdm

## Cell 4: final output (1 stream records collapsed)
    LorenzLiftMulti.npz found

## Cell 6: final output (2 stream records collapsed)
    holdout  mean +0.010943  std 1.001611
    PASS: train is exact 0/1; val/test/holdout are close but not exact -- no split's own
    statistics leaked into the constants used to build it.

## Cell 7: figure -> notebook_results/cell007_fig1.png
## Cell 8: figure -> notebook_results/cell008_fig2.png
## Cell 8: figure -> notebook_results/cell008_fig3.png
## Cell 9: figure -> notebook_results/cell009_fig4.png
## Cell 9: final output (3 stream records collapsed)
    observation channels: 30,  global range [-4.06, 3.36]  (pooled across 8 series)

## Cell 10: figure -> notebook_results/cell010_fig5.png
## Cell 13: final output (1 stream records collapsed)
    device cpu, torch threads 8
    60 epochs/model, marks (5, 10, 20, 40, 60), seeds [0, 1, 2], phi [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    110.4 steps per Lyapunov time -> horizon 550 = 5.0 LT, climate 5500 = 50 LT

## Cell 14: final output (1 stream records collapsed)
    naive noise**2                                : 0.002500
    closed-form noise**2/(1+noise**2)              : 0.002494
    NOISE_FLOOR is a measured quantity, distinct from both closed-form estimates.

## Cell 17: final output (1 stream records collapsed)
    forecast windows (64, 64, 30) -> (64, 550, 30)
    climate windows  (32, 64, 30) -> (32, 5500, 30)
    scale 1.0016   bounds -4.06 .. 3.36

## Cell 18: final output (1 stream records collapsed)
    27840 training windows of 64+20 from 8 series, 64 val windows
    XFull (27840, 84, 30)  (281 MB)

## Cell 20: final output (2 stream records collapsed)
    Stage A 111,039  Stage B 13,443

## Cell 26: figure -> notebook_results/cell026_fig6.png
## Cell 26: final output (36027 stream records collapsed)
    ours B (seed 2): 100%|██████████| 60/60 [06:13<00:00,  6.23s/ep, fcst=1.38e-01, val_recon=4.28e-03, val_fcst=9.99e-03]
    ours B (seed 2): 100%|██████████| 60/60 [06:13<00:00,  6.23s/ep, fcst=1.38e-01, val_recon=4.28e-03, val_fcst=9.99e-03]saved ours_seed2.pt
    stage A 60 epochs, stage B 60 epochs, 524.7s total

## Cell 28: figure -> notebook_results/cell028_fig7.png
## Cell 28: figure -> notebook_results/cell028_fig8.png
## Cell 28: final output (3 stream records collapsed)
    stage B fcst: 0.18675
    stage B val_fcst: 0.01483
    stage B val_recon: 0.00343

## Cell 31: figure -> notebook_results/cell031_fig9.png
## Cell 31: final output (2 stream records collapsed)
    test recon MSE 0.00357   (pooled across 8 series, floor 0.00249)
    test R2        0.9964   (pooled across 8 series)

## Cell 33: figure -> notebook_results/cell033_fig10.png
## Cell 33: figure -> notebook_results/cell033_fig11.png
## Cell 33: final output (3 stream records collapsed)
    worst channel R2 0.9941 (channel 8)

## Cell 35: figure -> notebook_results/cell035_fig12.png
## Cell 35: figure -> notebook_results/cell035_fig13.png
## Cell 36: figure -> notebook_results/cell036_fig14.png
## Cell 36: final output (2 stream records collapsed)
    |b| range [-20.21, 14.39]  (pooled across 8 series, unbounded by design)

## Cell 39: final output (522179 stream records collapsed)
    phi extra seeds: 100%|██████████| 3/3 [3:19:42<00:00, 3983.53s/it]
    phi extra seeds: 100%|██████████| 3/3 [3:19:42<00:00, 3994.02s/it]phi 0.5 seed 2 epoch 0060/60 | recon 2.279e-03 | fcst 3.198e-03 | val_recon 2.185e-03 | val_fcst 2.863e-03
    saved joint_phi0_5_seed2.pt

## Cell 40: figure -> notebook_results/cell040_fig15.png
## Cell 40: figure -> notebook_results/cell040_fig16.png
## Cell 42: figure -> notebook_results/cell042_fig17.png
## Cell 42: final output (7 stream records collapsed)
    sigmoid phi=0.5: 100%|██████████| 60/60 [47:11<00:00, 47.19s/ep, recon=2.32e-03, fcst=3.21e-03, val_recon=2.24e-03, val_fcst=3.01e-03]
    sigmoid phi=0.5: 100%|██████████| 60/60 [47:11<00:00, 47.20s/ep, recon=2.32e-03, fcst=3.21e-03, val_recon=2.24e-03, val_fcst=3.01e-03]sigmoid phi=0.5 epoch 0060/60 | recon 2.323e-03 | fcst 3.207e-03 | val_recon 2.241e-03 | val_fcst 3.013e-03
    saved sigmoid_phi0_5_seed2.pt

## Cell 44: figure -> notebook_results/cell044_fig18.png
## Cell 44: final output (30911 stream records collapsed)
    pinn rho=26 (seed 0): 100%|██████████| 60/60 [08:22<00:00,  8.38s/ep, recon=2.80e-02, physics=1.06e-01, val_recon=1.39e-02, val_fcst=5.04e-02]
    pinn rho=26 (seed 0): 100%|██████████| 60/60 [08:22<00:00,  8.38s/ep, recon=2.80e-02, physics=1.06e-01, val_recon=1.39e-02, val_fcst=5.04e-02]
    saved pinn_rho26_seed0.pt

## Cell 46: final output (1 stream records collapsed)
    eval horizon is longer than the training rollout by construction -- VPT/NRMSE@5LT
    measure genuine free-running extrapolation, not a length mismatch bug.
    final: val_recon 0.00365  physics 0.0369  val_fcst 0.0138  (pre-fix: 0.0037 / 0.031 / 17.6-ish and rising)

## Cell 47: figure -> notebook_results/cell047_fig19.png
## Cell 47: figure -> notebook_results/cell047_fig20.png
## Cell 47: final output (3 stream records collapsed)
    pinn rho=28 | 60 epochs | val_recon 0.00365 | physics 0.0369
    pinn rho=26 | 60 epochs | val_recon 0.01387 | physics 0.1059

## Cell 49: final output (15 stream records collapsed)
    phi=0.75               epoch  40-> 60  val_fcst 0.0035->0.0028  (+19.2%)  still moving
    phi=0.9                epoch  40-> 60  val_fcst 0.0033->0.0029  (+12.3%)  still moving
    phi=1                  epoch  40-> 60  val_fcst 0.0031->0.0028  (+8.0%)  still moving

## Cell 51: figure -> notebook_results/cell051_fig21.png
## Cell 53: final output (39803 stream records collapsed)
    phi=0, latent=16 (the actual Model B row above): recon MSE 0.00135  x floor 0.54345
    phi=0, latent=3  (this diagnostic, capacity-matched): recon MSE 0.00237  x floor 0.95188
    STILL below the floor with latent=3 -- capacity is NOT the explanation, something else is; treat the sub-floor result as unresolved.

## Cell 55: figure -> notebook_results/cell055_fig22.png
## Cell 56: figure -> notebook_results/cell056_fig23.png
## Cell 59: figure -> notebook_results/cell059_fig24.png
## Cell 59: figure -> notebook_results/cell059_fig25.png
## Cell 59: figure -> notebook_results/cell059_fig26.png
## Cell 60: figure -> notebook_results/cell060_fig27.png
## Cell 63: figure -> notebook_results/cell063_fig28.png
## Cell 63: final output (2 stream records collapsed)
    C outside [0,1] over 5500 steps x 32 rollouts: 0.000000
    C saturated (outside [0.02, 0.98]): 0.2943
    |b| max during rollout: 20.09

## Cell 64: figure -> notebook_results/cell064_fig29.png
## Cell 67: final output (11 stream records collapsed)
    climate:  86%|████████▌ | 6/7 [00:21<00:03,  3.35s/it]
    climate: 100%|██████████| 7/7 [00:24<00:00,  3.23s/it]
    climate: 100%|██████████| 7/7 [00:24<00:00,  3.53s/it]

## Cell 68: figure -> notebook_results/cell068_fig30.png
## Cell 69: figure -> notebook_results/cell069_fig31.png
## Cell 70: figure -> notebook_results/cell070_fig32.png
## Cell 73: figure -> notebook_results/cell073_fig33.png
## Cell 73: final output (2 stream records collapsed)
    phi=0.75                 142.8s
    phi=0.9                  202.4s
    phi=1                    569.9s

## Cell 75: figure -> notebook_results/cell075_fig34.png
## Cell 77: figure -> notebook_results/cell077_fig35.png
## Cell 77: final output (7 stream records collapsed)
    ours joint: 100%|██████████| 60/60 [1:20:24<00:00, 80.41s/ep, recon=2.56e-03, fcst=3.07e-03, val_recon=2.50e-03, val_fcst=3.12e-03]
    ours joint: 100%|██████████| 60/60 [1:20:24<00:00, 80.41s/ep, recon=2.56e-03, fcst=3.07e-03, val_recon=2.50e-03, val_fcst=3.12e-03]ours joint epoch 0060/60 | recon 2.561e-03 | fcst 3.070e-03 | val_recon 2.501e-03 | val_fcst 3.122e-03
    saved oursjoint_seed0.pt

## Cell 78: figure -> notebook_results/cell078_fig36.png
## Cell 78: figure -> notebook_results/cell078_fig37.png
## Cell 80: final output (2 stream records collapsed)
    Every trained row: 124,482 trainable params, 60 epochs over its own data, 8 CPU threads. seeds/epochs columns above make explicit which rows are 1-seed points (persistence/climatology have neither -- not trained at all).

## Cell 83: figure -> notebook_results/cell083_fig38.png
## Cell 83: final output (17 stream records collapsed)
    saved sigmoid_phi0_5_seed0.pt
    noise sweep: 100%|██████████| 2/2 [3:41:36<00:00, 6630.83s/it]
    noise sweep: 100%|██████████| 2/2 [3:41:36<00:00, 6648.09s/it]noise=0.3 done

## Cell 87: final output (1 stream records collapsed)
    noise=0.15: Ours 0.6342  best phi 0.7339  gap -0.0997
    noise=0.3: Ours 0.4530  best phi 0.5980  gap -0.1450
    DIRECTION: the gap NARROWS as noise increases (monotonically, 0.05 -> 0.15 -> 0.30).

## Cell 89: final output (3 stream records collapsed)
    paper_pareto.png
    paper_horizon.png
    paper figures written to C:\Users\ArpitGoel\Documents\GitHub\Bounded and unbounded latent decoupling for forecast and reconstruction quality\Paper\figs
