
"""
这个是用来当检测到程序结束后，自动拉起一个进程运行

"""
import subprocess

import os
import time
from fitsearch.utils import getGPUs

def check_pid(pid):
    # https://stackoverflow.com/a/568285
    """ Check For the existence of a unix pid. """
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    else:
        return True

watch_gpus = [2,3,4]
cmd = 'python train_ee.py -d oace05E+ --lr 1e-5 -b 32 -n 100 -a 1 --cross_dim 150 --biaffine_size 300 --model_name microsoft/deberta-v3-large --cross_depth 0  --drop_s1_p 0 --empty_rel_weight 0.1 --use_ln 1 --use_s2 1'
assert ','.join(map(str, watch_gpus))
print("Wait to run ", cmd)
while True:
    gpus = getGPUs()
    flags = []
    for idx in watch_gpus:
        if gpus[idx].memoryUsed < 100:
            flags.append(True)
        else:
            flags.append(False)
    if all(flags):
        break
    time.sleep(10)

subprocess.check_call(cmd, shell=True)