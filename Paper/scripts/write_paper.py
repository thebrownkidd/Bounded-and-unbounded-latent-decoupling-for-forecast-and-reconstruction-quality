import io, json

SP = ("C:/Users/ARPITG~1/AppData/Local/Temp/claude/c--Users-ArpitGoel-Documents-GitHub-"
      "Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality/"
      "cdd20b34-1bb7-4cf8-a41f-3ee2fc81363c/scratchpad/")
J = json.load(open(SP + "paper_numbers.json"))
Rt = json.load(open(SP + "paper_ratios.json"))
R, D, P, CL = J["results"], J["data"], J["params"], J["climate_50LT"]
BD, PT = J["boundedness"], J["passthrough"]
FLOOR = D["noise_floor_mse"]


def f(x, n=4):
    return f"{x:.{n}f}"


PROF = J["profile"]["by_model"]
LEADS = J["profile"]["leads_LT"]
I1 = LEADS.index(1.0)
ACC_T = J["profile"]["acc_threshold"]


def row(key, label, seeds):
    r = R[key]
    p = PROF[key]
    rec = "--" if r.get("recon_mse") is None else f(r["recon_mse"], 5)
    fl = "--" if r.get("recon_over_floor") is None else f(r["recon_over_floor"], 3)
    sd = r.get("vpt_std", 0.0)
    vpt = f(r["vpt"], 3) + (f"\\,$\\pm$\\,{sd:.3f}" if seeds > 1 else "")
    return (f"{label} & {seeds if seeds else '--'} & {rec} & {fl} & {vpt} & "
            f"{f(r['fcst_nrmse_full'], 3)} & {100*r['divergence']:.1f}\\% & "
            f"{f(p['mae'][I1], 3)} & {f(p['acc_horizon_LT'], 2)} \\\\")


def profrow(key, label, metric, fmt=3):
    return f"{label} & " + " & ".join(f(v, fmt) for v in PROF[key][metric]) + r" \\"


PROF_ORDER = [
    ("Ours (three-phase)", r"\textbf{Ours}"), ("phi=0", r"$\phi$=0"),
    ("phi=0.1", r"$\phi$=0.1"), ("phi=0.25", r"$\phi$=0.25"),
    ("phi=0.5", r"$\phi$=0.5"), ("phi=0.75", r"$\phi$=0.75"),
    ("phi=0.9", r"$\phi$=0.9"), ("phi=1", r"$\phi$=1"),
    ("AEGRU+sigmoid", "AEGRU+sigmoid"), ("PINN (true physics)", "PINN"),
    ("PINN (rho=26)", r"PINN ($\rho$=26)"), ("phi=0, latent=3", r"$\phi$=0, lat.\ 3"),
    ("persistence", "persistence"), ("climatology", "climatology"),
]
LEADHDR = " & ".join(f"{t:g}" for t in LEADS)
PROF_MAE = "\n".join(profrow(k, l, "mae") for k, l in PROF_ORDER)
PROF_ACC = "\n".join(profrow(k, l, "acc") for k, l in PROF_ORDER)
PROF_SS = "\n".join(profrow(k, l, "ss_persist", 2) for k, l in PROF_ORDER)
PROF_VR = "\n".join(profrow(k, l, "var_ratio") for k, l in PROF_ORDER)


TABLE = "\n".join([
    row("Ours (three-phase)", r"\textbf{Ours (three-phase)}", 3),
    row("Ours (one-stage)", "Ours (one-stage)", 1),
    r"\midrule",
    row("phi=0.1", r"$\phi$=0.1", 3),
    row("phi=0.5", r"$\phi$=0.5", 3),
    row("phi=0.25", r"$\phi$=0.25", 3),
    row("AEGRU+sigmoid", "AEGRU+sigmoid (D)", 3),
    row("PINN (true physics)", "PINN (true physics)", 3),
    row("phi=1", r"$\phi$=1", 1),
    row("phi=0", r"$\phi$=0", 1),
    r"\midrule",
    row("persistence", "persistence", 0),
    row("climatology", "climatology", 0),
])

CAP = "\n".join(
    f"{r['model'].replace('phi=', '$\\phi$=').replace('Ours (three-phase)', r'\textbf{Ours}')} & "
    f"{r['d']} & {f(r['over_floor'],3)} & {f(r['bound'],3)} & {f(r['over_own_optimum'],3)} \\\\"
    for r in Rt["recon_capacity_table"])

CLIMT = "\n".join(
    f"{k.replace('phi=', '$\\phi$=').replace('Ours (three-phase)', r'\textbf{Ours}')} & "
    f"{f(v['wasserstein'],4)} & {f(v['wasserstein_max'],4)} & {f(v['spectrum_log_err'],3)} \\\\"
    for k, v in CL.items())

PTT = "\n".join(
    f"{k.replace('phi=', '$\\phi$=').replace('Ours (three-phase)', r'\textbf{Ours}')} & "
    f"{v['d']} & {f(v['passthrough'],3)} & {f(v['d']/D['n_obs'],3)} & {f(3/D['n_obs'],3)} \\\\"
    for k, v in PT.items())

TEX = r"""\documentclass{article}

\usepackage[dblblindworkshop]{neurips_2026}

\workshoptitle{Foundation Models for Temporal Systems (FMTS)}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{float}

\title{Is the Reconstruction--Forecasting Tradeoff a Frontier\\or an Operating Point?}

\author{Arpit Goel\\ TwinSim Labs \\ arpit@thebrownkid.in}

\begin{document}

\maketitle

\begin{abstract}
Autoencoder based temporal models are usually trained on a weighted sum of a
reconstruction loss and a forecasting loss, with the weight tuned per dataset.
This assumes the balance between the two objectives is a real frontier. We ask
whether it is instead a bad operating point that comes from forcing one latent
to do both jobs. On a controlled chaotic benchmark (Lorenz-63 lifted to
__NOBS__ noisy channels, known intrinsic dimension __K__) we split the
representation into an unbounded __K__-dimensional carrier read only by the
forecaster and a sigmoid bounded __A__$\times$__B__ carrier read only by the
decoder, trained in three phases so the forecast gradient can never reach the
decoder. At matched parameters, matched epochs and matched seeds it reaches a
valid prediction time of __VPT__ Lyapunov times against __RIVVPT__ for the best
setting of the knob, __VPTX__ times longer. No baseline in the study beats
climatology at 5 Lyapunov times and ours does, with no rollout ever leaving the
training range. An ablation separates the two ingredients. We also explain a result that
looks impossible: __NBELOW__ of the baselines reconstruct better than the
measured noise floor, which turns out to be copying rather than denoising, and
the models that copy most forecast worst.
\end{abstract}

\section{Introduction}

A large body of work trains one encoder to serve two objectives at once. It has
to reconstruct the present and forecast the future. The standard way to do this
is a weighted loss,
$\mathcal{L} = (1-\phi)\mathcal{L}_{\mathrm{rec}} + \phi\mathcal{L}_{\mathrm{fcst}}$,
with $\phi$ tuned per dataset. Coupled Attention Networks \citep{coupledattn}
call their weights ``a tradeoff between the prediction and reconstruction
models''. CAAE \citep{caae} names the ``objective mismatch between training for
reconstruction fidelity and testing for discriminative decisions''.
Reconstruction plus prediction hybrids fuse both objectives onto one shared
latent, some with explicit weights \citep{memaae} and some by plainly summing
the two losses \citep{hyvae}. SSP \citep{ssp} chases the same separation by
spectral truncation, ``decoupling reconstruction fidelity from rollout
regularity''. In all of this work the tradeoff is something you tune. Almost
nobody asks whether the tuning is needed at all.

The ingredients for asking already exist, but separately. Bounded latents are
routine in representation learning \citep{fsq}. Koopman style forecasters
constrain the latent \emph{dynamics} to buy long horizon stability
\citep{azencot,koopa}. Neither line splits the two objectives across separate
carriers, and the Koopman forecasters still run a single latent through both.

\paragraph{What we are not claiming.}
Rate distortion theory \citep{shannon,berger} proves a frontier between rate and
distortion for a source, and the information bottleneck \citep{tishby} extends
this to predictive representations. We are not claiming to beat any such bound.
Rate distortion constrains distortion \emph{at a fixed rate} and says nothing
about which point inside the achievable region an architecture lands on. Our
claim is only the second one. At fixed capacity and fixed compute, a shared
latent under a weighted loss does not reach the best available point, and
separating the carriers moves it there.

\paragraph{Contributions.}
(i) A capacity matched and epoch matched test of whether the tradeoff is a
frontier or an operating point, on a system of known intrinsic dimension. (ii) A
three phase schedule with a bounded and an unbounded carrier that reaches
__VPTX__ times the forecast horizon of the best weighted loss setting, while
reconstructing at __FLOORX__ times the measured noise floor and never leaving
the training range. (iii) An explanation of why wide latent baselines appear to
reconstruct below the noise floor.

\section{Method}

Write $x_t \in \mathbb{R}^{n}$ for the observation at time $t$. The model has
six parts. An encoder $E$, a bounded teacher $r$, a decoder $D$, a compressor
$f$, a bounded map $m$, and a forecaster $g$. The inference path is

\begin{equation}
x_t \xrightarrow{\;E\;} h_t \xrightarrow{\;f\;} b_t \in \mathbb{R}^{k}
\xrightarrow{\;m\;} C_t \in [0,1]^{__A__ \times __B__} \xrightarrow{\;D\;} \hat{x}_t ,
\qquad b_{t+1} = g(b_t).
\end{equation}

The carrier $b$ is unbounded and only the forecaster reads it. The carrier $C$
is bounded and only the decoder reads it. $C$ stays inside $[0,1]$ because $m$
ends in a sigmoid, and that holds for a rollout of any length however far $b$
drifts, because it is a property of the function class and not of the optimiser.

\paragraph{Three phase training.}
The two carriers are separated by the schedule, not only by the wiring. Each
phase freezes everything it does not train.

\emph{Phase 1} trains $E$, $r$ and $D$ as a bounded autoencoder, using
$E(x_t)=h_t$, $r(h_t)=C_t$ and $D(C_t)=\hat{x}_t$ with
$\mathcal{L}_1 = \mathrm{MSE}(\hat{x}_t, x_t)$. The decoder never sees the
narrow carrier during its own training.

\emph{Phase 2} freezes $E$, $r$ and $D$, and trains $f$, $g$ and $m$ with
$f(h_t)=b_t$, $g(b_t)=b_{t+1}$ and $m(b_{t+1})=\hat{C}_{t+1}$. The loss lives in
$C$ space against the frozen phase 1 code,
$\mathcal{L}_2 = \mathrm{MSE}(C_{t+1}, \hat{C}_{t+1})$. The decoder is not used
at all here. The target $C_{t+1}=r(h_{t+1})$ is a fixed tensor, so $f$ cannot
escape by collapsing. A loss in $b$ space against the encoder's own $b$ would
collapse instead, because a constant $f$ and an identity $g$ drive it to zero
while learning nothing.

\emph{Phase 3} freezes $E$, $f$, $g$ and $r$, and fine tunes $m$ and $D$ with
$m(b_t)=\hat{C}_t$ and $D(\hat{C}_t)=\hat{x}_t$, under
$\mathcal{L}_3 = \mathrm{MSE}(\hat{C}_t, C_t) + \mathrm{MSE}(\hat{x}_t, x_t)$.

The decoupling is exact and not approximate. In phase 3 the forecaster is
frozen, so $\partial \mathcal{L}_3 / \partial \theta_g \equiv 0$. In phase 2 the
decoder is frozen and never even runs. No gradient from a forecast objective
ever reaches $D$.

\section{Experimental setup}

\paragraph{Data.} Lorenz-63 \citep{lorenz} at $dt=__DT__$, lifted through a
frozen random tanh map to __NOBS__ channels with additive noise, so the
intrinsic dimension is known to be __K__. We use __NSER__ training trajectories
of __NSTEPS__ steps each, split 70/15/15 in time and pooled, giving __POOLED__
timesteps. Every scaling constant is fit on that pooled training portion only. A
further __NHOLD__ trajectories are held out completely and never split, and all
reported numbers come from windows cut from those. The measured noise floor is
__FLOOR__ in the final standardised scale.

\paragraph{Metrics.} Reconstruction MSE on held out windows, and its ratio to
the noise floor. Valid prediction time (VPT), the lead time at which normalised
error first crosses __THRESH__, in Lyapunov times \citep{vlachas}. Normalised
error over the full 5 Lyapunov time horizon. Divergence, the fraction of
rollouts that leave the training range. One Lyapunov time is __SPL__ steps.

\paragraph{Protocol.} Every model gets __TARGET__ trainable parameters, matched
by bisecting the baseline hidden width, which lands within __PCTOFF__\% . Every
parameter group gets __EPOCHS__ passes over its own training data. Epochs are
used instead of wall clock because epoch counts reproduce on any machine and
wall clock does not. Three seeds where stated. Baselines are the weighted loss
autoencoder plus GRU swept over $\phi$, a sigmoid bounded shared latent control
that isolates boundedness from the split, and a physics informed model handed
the exact Lorenz equations.

\section{Results}

\begin{table}[t]
\caption{Held out results, sorted by forecast horizon. Lower is better for
reconstruction MSE, error at 5 LT and divergence. Higher is better for VPT.
Every trained row has __TARGET__ parameters and __EPOCHS__ epochs per parameter
group. The $\times$floor column is reconstruction MSE divided by the measured
noise floor. MAE and ACC horizon come from the long horizon pass described in
Section~\ref{sec:skill}. ACC horizon is the lead time at which anomaly
correlation drops below __ACCT__.}
\label{tab:main}
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccccc}
\toprule
model & seeds & recon MSE & $\times$floor & VPT (LT) & err @ 5LT & diverg.
& MAE @ 1LT & ACC hor. \\
\midrule
__TABLE__
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\begin{minipage}[t]{0.49\textwidth}
  \centering\includegraphics[width=\linewidth]{figs/paper_pareto.png}
\end{minipage}\hfill
\begin{minipage}[t]{0.49\textwidth}
  \centering\includegraphics[width=\linewidth]{figs/paper_horizon.png}
\end{minipage}
\caption{Left: reconstruction against forecast horizon. The $\phi$ sweep traces
the weighted loss frontier and our split sits well above all of it. Right:
normalised error against lead time on a log scale. Ours is the only curve that
stays under climatology across the whole horizon.}
\label{fig:main}
\end{figure}

\paragraph{The forecast gap is large.}
Our split reaches VPT __VPT__ Lyapunov times, __VPTSTEPS__ steps, against
__RIVVPT__ for the best knob setting at $\phi$=0.1, so __VPTX__ times longer.
Turning the knob does not close the gap, because VPT peaks at $\phi$=0.1 and
then falls. At 5 Lyapunov times ours reports __N5__ against __CLIM5__ for
climatology, and no baseline in the study beats predicting the mean at all.

\paragraph{What the architecture buys and what the schedule buys.}
The same six modules trained end to end in one stage reach VPT __OJVPT__ against
__VPT__, so the schedule is worth __OJX__ times on horizon. It buys nothing on
reconstruction (__OJFLOOR__ against __FLOORX__ times the floor) or on stability
(both __DIV__\%), which the architecture already supplies. Both ingredients
matter, because the one stage run alone beats the best knob setting by
__OJVSPHI__ times.

\paragraph{Forecast skill agrees with VPT, on a different axis.}
\label{sec:skill}
VPT reads one error threshold, so we also profiled MAE, anomaly correlation
(ACC), skill against persistence and forecast variance out to 50 Lyapunov times
(Appendix~\ref{app:profile}). At one Lyapunov time ours reports MAE __MAE1__
against __MAE1R__ for the best baseline, and ACC stays above __ACCT__ out to
__ACCH__ Lyapunov times against __ACCHR__, a factor of __ACCHX__. ACC scores
whether the pattern is right and VPT whether the error is small, so this is an
independent check. Ours also holds variance ratio __VR50__ at 50 Lyapunov times
while __NEXPL__ knob settings blow past twice the true variance.

\paragraph{Stability is a guarantee, not a tuning result.}
Over __CSTEPS__ free running steps no entry of $C$ ever leaves $[0,1]$ while
$\lvert b \rvert$ reaches __BMAX__, which is the bounded carrier doing what the
construction promises. Attractor fidelity agrees, with Wasserstein __WOURS__
against __WPINN__ for the PINN and __WPHI__ for $\phi$=0.1.

\section{Why some baselines land under the noise floor}

__NBELOW__ of the trained baselines reconstruct \emph{below} the measured noise
floor. That looks impossible, but it is not. The floor is the MSE you get by
outputting the clean signal, while reconstruction is scored as
$\mathrm{MSE}(\hat{x}, x)$ where $x$ is the \emph{noisy} observation and the
model takes $x$ as its input. So the floor was never a lower bound here. The
identity map settles it. Set $\hat{x} = x$ and the score is exactly zero.

The clean signal lies on a $k$ dimensional manifold because it is a fixed
function of the Lorenz state, while the noise spreads over all $n$ channels. A
model with a $d$ dimensional bottleneck can carry $d$ of those noise directions
through instead of discarding them, so the achievable minimum is
\begin{equation}
\mathrm{MSE}_{\min} = \frac{n-d}{n} \times \mathrm{floor},
\end{equation}
not the floor itself. Figure~\ref{fig:capacity} in the appendix checks this
against optimal linear rank $d$ reconstruction of the same held out windows. At
the baseline latent width $d$=16 the measured value is __PCA16__ against a
predicted __PCA16P__.

We measure the copying directly. Perturb the input a little and see how much
survives to the output. A pure identity gives 1, a rank $d$ projection gives
$d/n$, and a model passing only the signal gives $k/n$. At $\phi$=0 we measure
__PT0__ against __PT0P__ for a rank 16 copy, while ours measures __PTOURS__
against __PTOURSP__ for a pure denoiser.

Two things follow. First, the reconstruction column is not comparable across
bottleneck widths, because wide models have a lower achievable floor. Normalised
by each model's own bound, ours is best of the trained models at __OWNOPT__
against __RIVOWN__ (Table~\ref{tab:capacity}). Second, copying and forecasting
conflict. Passthrough falls from __PT0__ at $\phi$=0 to __PT05__ at $\phi$=0.5
and __PT1__ at $\phi$=1, and $\phi$=0, which copies most and looks best on
reconstruction, is the worst forecaster in the study at VPT __PHI0VPT__ with
__PHI0DIV__\% divergence. The reconstruction win and the forecast collapse are
one fact measured twice.

\section{Limitations}

\textbf{One system.} Lorenz-63, one lift, one noise level. \textbf{Uneven
budget.} $m$ and $D$ train in two phases and see __EPOCHS2__ passes against
__EPOCHS__, and the one stage ablation gets __EPOCHS__ epochs against three
phases of __EPOCHS__, so part of the schedule gap may be compute.
\textbf{Uneven seeds.} Single seed rows report zero spread. \textbf{Constructed
baselines.} The $\phi$ sweep and the sigmoid control are the balancing recipe in
its plainest form, not named methods. \textbf{Fixed capacity.} The bounded
carrier is a fixed __A__$\times$__B__ grid we never varied.

\section{Conclusion}

At matched parameters, matched epochs and matched seeds, separating a bounded
decoder facing carrier from an unbounded forecaster facing one reaches an
operating point no setting of the weighted loss knob reaches: __VPTX__ times its
forecast horizon, reconstruction at __FLOORX__ times the noise floor, and no
rollout leaving the training range. Read together with the copying result, the
tradeoff looks less like a frontier and more like a symptom of asking one
carrier to do two jobs.

\bibliographystyle{plainnat}
\begin{thebibliography}{99}

\bibitem[Azencot et al.(2020)]{azencot}
O.~Azencot, N.~B. Erichson, V.~Lin, and M.~W. Mahoney.
\newblock Forecasting sequential data using consistent Koopman autoencoders.
\newblock In \emph{Proceedings of the 37th International Conference on Machine
  Learning (ICML)}, pages 475--485, 2020.

\bibitem[Berger(1971)]{berger}
T.~Berger.
\newblock \emph{Rate Distortion Theory: A Mathematical Basis for Data
  Compression}.
\newblock Prentice-Hall, Englewood Cliffs, NJ, 1971.

\bibitem[Cai et al.(2023)]{hyvae}
B.~Cai, S.~Yang, L.~Gao, and Y.~Xiang.
\newblock Hybrid variational autoencoder for time series forecasting.
\newblock \emph{Knowledge-Based Systems}, 281:111079, 2023.
\newblock arXiv:2303.07048.

\bibitem[Liu et al.(2023)]{koopa}
Y.~Liu, C.~Li, J.~Wang, and M.~Long.
\newblock Koopa: Learning non-stationary time series dynamics with Koopman
  predictors.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)},
  2023.
\newblock arXiv:2305.18803.

\bibitem[Lorenz(1963)]{lorenz}
E.~N. Lorenz.
\newblock Deterministic nonperiodic flow.
\newblock \emph{Journal of the Atmospheric Sciences}, 20(2):130--141, 1963.

\bibitem[Lu et al.(2026)]{ssp}
X.~Lu, Y.~Yuan, and J.~Shi.
\newblock Stable long-horizon PDE forecasting via latent structured spectral
  propagators.
\newblock \emph{arXiv preprint arXiv:2605.10154}, 2026.

\bibitem[Mentzer et al.(2024)]{fsq}
F.~Mentzer, D.~Minnen, E.~Agustsson, and M.~Tschannen.
\newblock Finite scalar quantization: VQ-VAE made simple.
\newblock In \emph{International Conference on Learning Representations (ICLR)},
  2024.
\newblock arXiv:2309.15505.

\bibitem[Shannon(1959)]{shannon}
C.~E. Shannon.
\newblock Coding theorems for a discrete source with a fidelity criterion.
\newblock \emph{IRE National Convention Record}, part 4, pages 142--163, 1959.

\bibitem[Tishby et al.(2000)]{tishby}
N.~Tishby, F.~C. Pereira, and W.~Bialek.
\newblock The information bottleneck method.
\newblock \emph{arXiv preprint physics/0004057}, 2000.

\bibitem[Vlachas et al.(2020)]{vlachas}
P.~R. Vlachas, J.~Pathak, B.~R. Hunt, T.~P. Sapsis, M.~Girvan, E.~Ott, and
  P.~Koumoutsakos.
\newblock Backpropagation algorithms and reservoir computing in recurrent neural
  networks for the forecasting of complex spatiotemporal dynamics.
\newblock \emph{Neural Networks}, 126:191--217, 2020.

\bibitem[Xia et al.(2024)]{coupledattn}
F.~Xia, X.~Chen, S.~Yu, M.~Hou, M.~Liu, and L.~You.
\newblock Coupled attention networks for multivariate time series anomaly
  detection.
\newblock \emph{IEEE Transactions on Emerging Topics in Computing},
  12:240--253, 2024.
\newblock arXiv:2306.07114.

\bibitem[Xiao et al.(2021)]{memaae}
Q.~Xiao, S.~Shao, and J.~Wang.
\newblock Memory-augmented adversarial autoencoders for multivariate
  time-series anomaly detection with deep reconstruction and prediction.
\newblock \emph{arXiv preprint arXiv:2110.08306}, 2021.

\bibitem[Xie et al.(2026)]{caae}
X.~Xie, K.~Liu, Y.~Wang, M.~Wu, H.~Zhang, and T.~Wan.
\newblock CAAE: Contrastive adversarial autoencoder for multivariate time
  series anomaly detection.
\newblock \emph{Pattern Recognition}, 2026.

\end{thebibliography}

\newpage
\appendix

\section{Capacity normalised reconstruction}

Reconstruction MSE is scored against the noisy observation, so a model with a
$d$ dimensional bottleneck has achievable minimum $(n-d)/n$ times the floor
rather than the floor itself. Table~\ref{tab:capacity} divides each model's
score by its own bound. On this view ours is the best of the trained models. The
capacity matched diagnostic, a $\phi$=0 baseline with its latent cut from 16 to
__K__, is included to show that the effect tracks width and not architecture. It
lands at __DIAGOWN__ times its own optimum, which is close to ours, and its
forecast collapses in exactly the way $\phi$=0 does.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figs/paper_capacity.png}
\caption{Left: optimal linear rank $d$ reconstruction of the held out windows
tracks the capacity bound $(n-d)/n$ almost exactly, and sits far under the
noise floor for any wide bottleneck. Right: measured passthrough, the fraction
of a random input perturbation that survives to the output. $\phi$=0 behaves
like a rank 16 copy of its input. Ours sits at the pure denoiser value, because
a __K__ dimensional carrier has no spare width to copy with.}
\label{fig:capacity}
\end{figure}

\begin{table}[H]
\caption{Reconstruction against each model's own capacity bound. $d$ is the
bottleneck width. The bound column is $(n-d)/n$.}
\label{tab:capacity}
\centering
\small
\begin{tabular}{lcccc}
\toprule
model & $d$ & $\times$floor & bound & $\times$ own optimum \\
\midrule
__CAP__
\bottomrule
\end{tabular}
\end{table}

\section{Input passthrough}

Passthrough is measured by perturbing the input with small isotropic noise and
recording the fraction of that perturbation which survives to the output,
averaged over 8 draws. A pure identity gives 1. A rank $d$ projection gives
$d/n$. A model whose output depends only on the __K__ dimensional signal gives
$k/n$.

\begin{table}[H]
\caption{Measured passthrough against the two reference behaviours.}
\label{tab:passthrough}
\centering
\small
\begin{tabular}{lcccc}
\toprule
model & $d$ & measured & rank $d$ copy & pure denoiser \\
\midrule
__PTT__
\bottomrule
\end{tabular}
\end{table}

\section{Long horizon attractor statistics}

Free running for __CSTEPS__ steps, which is __CLT__ Lyapunov times, on
__CROLL__ held out windows. Wasserstein distance is averaged over the observed
channels and the max column is the worst channel.

\begin{table}[H]
\caption{Attractor fidelity at __CLT__ Lyapunov times.}
\label{tab:climate}
\centering
\small
\begin{tabular}{lccc}
\toprule
model & Wasserstein & worst channel & log spectrum error \\
\midrule
__CLIMT__
\bottomrule
\end{tabular}
\end{table}

\section{Full lead time profile}
\label{app:profile}

Free running for __CSTEPS__ steps on __CROLL__ held out windows, scored at seven
lead times. MAE is in standardised observation units. ACC is anomaly correlation
against the training climatology, where 1 is perfect and 0 is no better than
predicting the mean. Skill is measured against persistence, so 0 means no better
than assuming nothing changes and negative means worse. The variance ratio is
forecast standard deviation over true standard deviation, so 1 is the right
amplitude, below 1 is collapsing toward the mean and above 1 is blowing up.

The variance ratio is worth reading alongside divergence, because the two
disagree in an informative way. $\phi$=0.1 holds amplitude __VR50R__ at 50
Lyapunov times and never leaves the training range, yet it scores worst of all
on attractor fidelity. It neither collapses nor explodes. It simply
decorrelates, which is a failure only ACC and the attractor statistics can see.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figs/paper_profile.png}
\caption{Lead time profile for a representative subset. Ours is the only model
whose anomaly correlation is still above __ACCT__ past one Lyapunov time, and it
holds the right variance for the whole run while $\phi$=0.5 grows past
$100\times$ the true amplitude.}
\label{fig:profile}
\end{figure}

\begin{table}[H]
\caption{MAE at each lead time, in Lyapunov times.}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
model & __LEADHDR__ \\
\midrule
__PROF_MAE__
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\caption{Anomaly correlation at each lead time.}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
model & __LEADHDR__ \\
\midrule
__PROF_ACC__
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\caption{Skill against persistence at each lead time. Large negative values mean
the forecast has left the attractor entirely.}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
model & __LEADHDR__ \\
\midrule
__PROF_SS__
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\caption{Variance ratio at each lead time. 1 is the right amplitude.}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
model & __LEADHDR__ \\
\midrule
__PROF_VR__
\bottomrule
\end{tabular}
\end{table}

\section{Reproducibility}

All numbers in this paper are produced by code from saved checkpoints, including
every ratio quoted in the text. The three phase schedule runs __EPOCHS__ epochs
per phase on __NSEEDS__ seeds. Phase 1 takes __T1__ seconds, phase 2 takes
__T2__ seconds and phase 3 takes __T3__ seconds on a single desktop CPU with 8
threads. Metrics are snapshotted at epochs __MARKS__ so the whole compute
scaling curve comes from one run rather than from separate runs at different
budgets.

\end{document}
"""

pt_ours = PT["Ours (three-phase)"]
O_VPT = R["Ours (three-phase)"]["vpt"]
sub = {
    "__NOBS__": str(D["n_obs"]), "__K__": str(D["k"]),
    "__A__": str(D["grid"][0]), "__B__": str(D["grid"][1]),
    "__DT__": f"{D['dt']}",
    "__NSER__": str(D["n_train_series"]), "__NSTEPS__": f"{20000:,}",
    "__POOLED__": f"{D['pooled_train_steps']:,}", "__NHOLD__": str(D["holdout_shape"][0]),
    "__FLOOR__": f"{FLOOR:.6f}", "__THRESH__": f(D["threshold"], 1),
    "__SPL__": f(D["steps_per_lyapunov"], 1),
    "__TARGET__": f"{P['target']:,}", "__PCTOFF__": f(Rt["param_match"]["max_abs_pct_off"], 2),
    "__EPOCHS__": "60", "__EPOCHS2__": "120", "__NSEEDS__": "3",
    "__VPT__": f(Rt["ours_vpt"], 3), "__VPTSTEPS__": f"{Rt['ours_vpt_steps']:.0f}",
    "__RIVVPT__": f(Rt["best_rival_vpt"], 3), "__VPTX__": f(Rt["ours_vpt_over_best_rival"], 2),
    "__N5__": f(Rt["ours_nrmse5"], 3), "__CLIM5__": f(Rt["clim_nrmse5"], 3),
    "__N5X__": f(Rt["ours_nrmse5_over_clim"], 2),
    "__DIV__": f"{100*Rt['ours_divergence']:.0f}",
    "__RIVDIV__": f"{100*R['phi=0.1']['divergence']:.1f}",
    "__FLOORX__": f(Rt["ours_over_floor"], 3),
    "__OWNOPT__": f(Rt["ours_over_own_optimum"], 3),
    "__RIVOWN__": f(Rt["best_recon_rival"]["over_own_optimum"], 3),
    "__NBELOW__": str(Rt["n_models_below_floor"] - 1),
    "__PCA16__": f(Rt["pca_check"]["16"]["measured"], 3),
    "__PCA16P__": f(Rt["pca_check"]["16"]["predicted"], 3),
    "__PT0__": f(PT["phi=0"]["passthrough"], 3),
    "__PT0P__": f(16 / D["n_obs"], 3),
    "__PT05__": f(PT["phi=0.5"]["passthrough"], 3),
    "__PT1__": f(PT["phi=1"]["passthrough"], 3),
    "__PTOURS__": f(pt_ours["passthrough"], 3),
    "__PTOURSP__": f(3 / D["n_obs"], 3),
    "__PHI0VPT__": f(R["phi=0"]["vpt"], 3),
    "__PHI0DIV__": f"{100*R['phi=0']['divergence']:.0f}",
    "__PINN26F__": f(R["PINN (rho=26)"]["recon_over_floor"], 2),
    "__CSTEPS__": f"{D['climate_steps']:,}", "__CROLL__": str(D["climate_windows"][0]),
    "__CLT__": f"{D['climate_LT']:.0f}",
    "__CMIN__": f"{BD['C_min']:.2e}", "__CMAX__": f(BD["C_max"], 6),
    "__BMAX__": f(BD["b_absmax"], 1),
    "__WOURS__": f(CL["Ours (three-phase)"]["wasserstein"], 4),
    "__WPINN__": f(CL["PINN (true physics)"]["wasserstein"], 4),
    "__WPHI__": f(CL["phi=0.1"]["wasserstein"], 3),
    "__DIAGOWN__": f([r for r in Rt["recon_capacity_table"]
                      if r["model"] == "phi=0, latent=3"][0]["over_own_optimum"], 3),
    "__T1__": f"{J['schedule']['phase1']['total_time_s']:.0f}",
    "__T2__": f"{J['schedule']['phase2']['total_time_s']:.0f}",
    "__T3__": f"{J['schedule']['phase3A']['total_time_s']:.0f}",
    "__MARKS__": ", ".join(str(m) for m in J["schedule"]["phase1"]["marks"]),
    "__TABLE__": TABLE, "__CAP__": CAP, "__PTT__": PTT, "__CLIMT__": CLIMT,
    # -- lead time profile
    "__ACCT__": f(ACC_T, 1),
    "__MAE1__": f(Rt["mae1_ours"], 3),
    "__MAE1R__": f(Rt["mae1_best_rival"]["value"], 3),
    "__MAE1X__": f(Rt["mae1_ratio"], 2),
    "__ACCH__": f(Rt["acc_horizon_ours"], 2),
    "__ACCHR__": f(Rt["acc_horizon_best_rival"]["value"], 2),
    "__ACCHX__": f(Rt["acc_horizon_ratio"], 2),
    "__VR50__": f(Rt["varratio_ours_50"], 3),
    "__VR50R__": f(Rt["varratio_phi01_50"], 3),
    "__NEXPL__": str(Rt["n_exploding_50"]),
    "__OJVPT__": f(R["Ours (one-stage)"]["vpt"], 3),
    "__OJFLOOR__": f(R["Ours (one-stage)"]["recon_over_floor"], 3),
    "__OJX__": f(O_VPT / R["Ours (one-stage)"]["vpt"], 2),
    "__OJVSPHI__": f(R["Ours (one-stage)"]["vpt"] / R["phi=0.1"]["vpt"], 2),
    "__LEADHDR__": LEADHDR,
    "__PROF_MAE__": PROF_MAE, "__PROF_ACC__": PROF_ACC,
    "__PROF_SS__": PROF_SS, "__PROF_VR__": PROF_VR,
}
for k, v in sub.items():
    TEX = TEX.replace(k, v)

assert "__" not in TEX.replace("\\_\\_", ""), [w for w in TEX.split() if "__" in w][:5]

io.open("Paper/fmts2026.tex", "w", encoding="utf-8", newline="").write(TEX)
print("wrote Paper/fmts2026.tex")
print("em dashes:", TEX.count("---"), "| semicolons:", TEX.count(";"))
