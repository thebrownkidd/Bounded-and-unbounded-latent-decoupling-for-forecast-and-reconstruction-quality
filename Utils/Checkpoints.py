"""Skip-if-trained caching for the comparison notebooks.

Training twelve-odd models at 120s each is the bulk of a notebook run. This
lets a training function check a directory first and load a finished model's
weights instead of re-training it, so re-running the notebook after an
evaluation-cell edit doesn't mean sitting through Model A-C again.

Only state_dict + history are persisted, not the whole nn.Module -- pickling a
live model ties the checkpoint to the exact class definition at save time,
which breaks the moment the architecture is edited. state_dict is just tensors
and survives that.
"""

import torch


def TrainOrLoad(Path, BuildFn, TrainFn, Device="cpu", Verbose=True):
    """Path -> (Model, Hist).

    BuildFn() -> a freshly constructed, untrained nn.Module.
    TrainFn(Model) -> Hist, training Model in place.

    If Path exists, construct the model and load its saved state_dict instead
    of calling TrainFn -- the constructor still runs, so architecture-specific
    setup (e.g. PhysicsLatentAE's buffers) stays consistent, but every gradient
    step is skipped. Otherwise train normally, then save.
    """
    Model = BuildFn()
    if Path.exists():
        Ckpt = torch.load(Path, map_location=Device, weights_only=False)
        Model.load_state_dict(Ckpt["state_dict"])
        if Verbose:
            print(f"loaded {Path.name}, skipping training")
        return Model, Ckpt["hist"]

    Hist = TrainFn(Model)
    Path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": Model.state_dict(), "hist": Hist}, Path)
    if Verbose:
        print(f"saved {Path.name}")
    return Model, Hist
