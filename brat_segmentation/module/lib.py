from path import Path
import tempfile
import os

import nibabel as nib
import numpy as np

import numpy as np


# basics
import torch
from tqdm import tqdm


from monai.networks.nets import BasicUNet
from monai.inferers import SlidingWindowInferer

# tta
from monai.transforms import RandGaussianNoised


# transforms
from monai.transforms import (
    Compose,
    LoadImageD,
    Lambdad,
    ToTensord,
    ScaleIntensityRangePercentilesd,
)

# dataloader
import monai
from monai.data import list_data_collate
from torch.utils.data import DataLoader


def vprint(*args):
    verbose = True
    if verbose:
        print(*args)


def create_nifti_seg(batch_element, onehot_model_outputs_CHWD, output_file):
    vprint("generate seg")
    # generate segmentation nifti
    one_hot = (
        (onehot_model_outputs_CHWD[0][:, :, :, :].sigmoid() > 0.5)
        .detach()
        .cpu()
        .numpy()
    )
    one_hot = one_hot.astype(int)

    wt = one_hot[0]
    tc = one_hot[1]
    et = one_hot[2]

    vprint(wt.shape)
    vprint(tc.shape)
    vprint(et.shape)

    multi_cold = np.zeros(wt.shape, dtype=np.uint8)
    # BraTS encoding:
    # 1: necrosis
    # 2: edema
    # 4: enhancing
    multi_cold[wt == 1] = 2  # fall back to edema
    multi_cold[tc == 1] = 1  # fall back to necrosis for tc
    multi_cold[et == 1] = 4  # enhancing tumor

    # get header and affine from T1
    T1 = nib.load(batch_element["t1"][0])

    segmentation_image = nib.Nifti1Image(multi_cold, T1.affine, T1.header)
    nib.save(segmentation_image, output_file)


def infer(
    input_dict,
    output_dict,
    options_dict=None,
):
    """runs inference for a BraTS style glioma segmentation model"""
    # S E T T I N G S
    # set device via environments!
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    multi_gpu = False  # keep this here please, even though it is not used now

    if options_dict is None:
        options_dict = {"workers": 0, "tta": True, "inference_batches": 20}
    else:
        tmp = {"workers": 0, "tta": True, "inference_batches": 20}
        tmp.update(options_dict)
        options_dict = tmp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_batches = options_dict.get("inference_batches", 20)
    print(
        "using",
        str(inference_batches),
        "inference batches; decrease this number when running out of memory!",
    )
    tta = options_dict.get("tta", True)
    inf_batch_length = 0  # set to 0 to loop through all batches
    segmentation = Path(os.path.abspath(output_dict["segmentation"]))
    workers = options_dict.get("workers", 0)

    # T R A N S F O R M S
    inference_transforms = Compose(
        [
            # << PREPROCESSING transforms >>
            LoadImageD(keys=["images"]),
            Lambdad(["images"], np.nan_to_num),
            # normalize intensities
            # NormalizeIntensityd(keys="images", channel_wise=True),
            # TODO currently we normalize on a batch basis, this might become problematic on gpus with very small gpus
            ScaleIntensityRangePercentilesd(
                keys=["images"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                relative=False,
            ),
            ToTensord(keys=["images"]),
        ]
    )

    # D A T A L O A D E R
    # get input data
    i_t1 = Path(os.path.abspath(input_dict["t1"]))
    i_t1c = Path(os.path.abspath(input_dict["t1c"]))
    i_t2 = Path(os.path.abspath(input_dict["t2"]))
    i_fla = Path(os.path.abspath(input_dict["fla"]))

    # dict
    exam_name = "brats_segmentation"
    images = [i_t1, i_t1c, i_t2, i_fla]

    infer_dicts = []

    theDict = {
        "patient": exam_name,
        "exam": exam_name,
        "t1": i_t1,
        "t1c": i_t1c,
        "t2": i_t2,
        "fla": i_fla,
        "images": images,
    }
    infer_dicts.append(theDict)

    # dataset and loader
    inf_ds = monai.data.Dataset(data=infer_dicts, transform=inference_transforms)
    inf_data_loader = DataLoader(
        inf_ds,
        batch_size=1,
        num_workers=workers,
        collate_fn=list_data_collate,
        shuffle=True,
    )

    # M O D E L
    # load checkpoint / best, dice or last
    # checkpoint_path = Path("checkpoints/checkpoint_best.pth.tar")
    # checkpoint_path = Path("checkpoints/checkpoint_last.pth.tar")
    cur_file_dir = os.path.dirname(os.path.realpath(__file__))
    chpath = options_dict.get("checkpoint", os.path.join(cur_file_dir, "checkpoints","checkpoint_dice.pth.tar"))
    checkpoint_path = Path(chpath)

    vprint("checkpoint exists?", os.path.exists(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = BasicUNet(
        dimensions=3,
        in_channels=4,
        out_channels=3,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
        act="mish",
    )

    # load cp
    model.load_state_dict(checkpoint["model_state"])

    # send model to device
    model = model.to(device)

    # I N F E R E R
    patch_size = (128, 128, 32)
    sw_batch_size = inference_batches
    overlap = 0.5

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )

    # R U N   I N F E R E N C E
    torch.cuda.empty_cache()

    with torch.no_grad():
        model.eval()
        # loop through batches

        for counter, data in enumerate(tqdm(inf_data_loader, 0)):
            if inf_batch_length != 0:
                if counter == inf_batch_length:
                    break

            # get the inputs and labels
            inputs = data["images"].float().to(device)

            # we compute it patch wise instead
            vprint("inputs:", inputs.shape)

            outputs = inferer(inputs, model)
            vprint("outputs.shape:", outputs.shape)

            # test time augmentations
            if tta == True:
                n = 1.0
                for _ in range(4):
                    # test time augmentations
                    _img = RandGaussianNoised(keys="images", prob=1.0, std=0.01)(data)[
                        "images"
                    ]

                    output = inferer(_img.to(device), model)
                    outputs = outputs + output
                    n = n + 1.0
                    for dims in [[2], [3]]:
                        flip_pred = inferer(
                            torch.flip(_img.to(device), dims=dims), model
                        )
                        output = torch.flip(flip_pred, dims=dims)
                        outputs = outputs + output
                        n = n + 1.0
                outputs = outputs / n
                vprint("outputs.shape:", outputs.shape)

            create_nifti_seg(
                batch_element=data,
                onehot_model_outputs_CHWD=outputs,
                output_file=segmentation,
            )

            # TODO consider saving model outputs
