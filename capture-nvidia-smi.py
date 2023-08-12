import subprocess
import time


def capture_metric(out_file):
    result = subprocess.run(["nvidia-smi | grep 300W"], shell=True, capture_output=True)

    metrics = []
    lines = result.stdout.decode("UTF-8").replace("|", "").replace("/", "").split("\n")
    for line in lines[:3]:
        val = line.split()
        metrics.append("{},{},{}".format(
            val[3].replace("W", ""),
            val[5].replace("MiB", ""),
            val[7].replace("%", ""))
        )
    with open(out_file, "a") as f:
        f.write(",".join(metrics))
        f.write("\n")


while True:
    capture_metric("exp/JAX_068_ds1_nerf_profile_2gpu_batch16384/gpu-metrics.txt")
    time.sleep(5)
