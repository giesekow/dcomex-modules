import torch
from path import Path


def shrinker(input_checkpoint):
    input_checkpoint = Path(input_checkpoint)

    old_cp = torch.load(input_checkpoint, map_location="cpu")

    cp_dir = Path("module/checkpoints")
    cp_file = cp_dir + "/" + input_checkpoint.name
    new_cp = {"model_state": old_cp["model_state"]}
    torch.save(new_cp, cp_file)


if __name__ == "__main__":
    raw_checkpoints = Path("module/raw_checkpoints")
    raw_cps = raw_checkpoints.files("*.tar")
    for cp in raw_cps:
        print("shrinking:", cp)
        shrinker(cp)
