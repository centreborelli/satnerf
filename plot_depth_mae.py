import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from metrics import dsm_pointwise_abs_errors
from eval_aoi import eval_aoi

def sorted_nicely(l):
    import re
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def plot_depth_mae(input_txt, logs_dir, results_dir, gt_dsm_path, last_epoch, output_png):

    gt_roi_path = gt_dsm_path.replace(".tif", ".txt")
    if not os.path.exists(gt_dsm_path):
        raise ValueError("Could not find {}".format(gt_dsm_path))
    if not os.path.exists(gt_roi_path):
        raise ValueError("Could not find {}".format(gt_roi_path))
    if not os.path.exists(input_txt):
        raise ValueError("Could not find {}".format(input_txt))

    gt_roi = np.loadtxt(gt_roi_path)
    with open(input_txt, "r") as f:
        content = f.readlines()
        run_ids = [x.strip() for x in content]

    for epoch_number in np.arange(1, last_epoch+1):
        for e in run_ids:
            eval_aoi(e, logs_dir, results_dir, epoch_number, checkpoints_dir=None)
        print("Epoch {} completed".format(epoch_number))

    # the lists with the evolution of the mae for each run will be stored in a dictionary
    print("\nComputing all mean absolute errors. This may take some minutes...")
    mae = {}
    for e in run_ids:
        mae[e] = []
        for n in np.arange(1, last_epoch+1):
            errors_from_all_val_ims = []
            dsm_paths = sorted_nicely(glob.glob(os.path.join(results_dir, "{}/dsm/*epoch{}.tif".format(e, n))))
            for d in dsm_paths:
                tmp1 = d.replace("/dsm/", "/dsm_crops/")
                tmp2 = d.replace("/dsm/", "/dsm_err/")
                abs_err = dsm_pointwise_abs_errors(d, gt_dsm_path, gt_roi, out_rdsm_path=tmp1, out_err_path=tmp2)
                errors_from_all_val_ims.append(np.nanmean(abs_err.ravel()))
            mae[e].append(np.mean(errors_from_all_val_ims))

    colors = ["blue", "firebrick", "magenta", "green", "grey", "darkorange"]

    plt.figure(figsize=(10, 5))
    labels = list(mae.keys())
    for i, (e, c) in enumerate(zip(run_ids, colors)):
        plt.plot(np.arange(1, len(mae[e])+1), mae[e], color=c)
        labels[i] += "    last: {:.2f} m".format(mae[e][-1])
    plt.xlabel('epochs')
    plt.ylabel('MAE [m]')
    plt.legend(labels)
    plt.savefig(output_png, bbox_inches='tight')
    print("... done! Output figure: {}".format(output_png))

if __name__ == '__main__':
    import fire
    fire.Fire(plot_depth_mae)








