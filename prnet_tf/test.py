from absl import app, flags
from absl.flags import FLAGS
import os
import numpy as np

from api import PRN
from modules.utils import load_yaml, set_memory_growth
from benchmark_aflw2000 import ana as ana_aflw2000, calc_nme as calc_nme_aflw2000
from benchmark_aflw import ana as ana_aflw, calc_nme as calc_nme_aflw

flags.DEFINE_string('cfg_path', './configs/prnet.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


def cal_nme_with_68kp(model, root, flist):
    pts68_fit_all = []
    null_index = []
    kpt = np.zeros((2, 68))
    for i, img_file in enumerate(flist):
        img_path = os.path.join(root, img_file)
        pos = model.process(img_path)
        if pos is None:
            pts68_fit_all.append(kpt)  # append the previous keypoint temporary work around
            null_index.append(i)
            continue

        kpt = model.get_landmarks(pos)
        kpt = kpt.transpose()
        kpt = kpt[:2, :]

        pts68_fit_all.append(kpt)

    return pts68_fit_all, null_index


def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    set_memory_growth()

    # load PRNet model
    cfg = load_yaml(FLAGS.cfg_path)
    model = PRN(cfg, is_dlib=True)

    # Calculate NME AFLW2000-3D
    root_aflw2000 = 'data/test.data/AFLW2000-3D_crop'
    img_list_aflw2000 = 'data/test.data/AFLW2000-3D_crop.list'
    with open(img_list_aflw2000) as f:
        flist_aflw2000 = [line.rstrip() for line in f]
    pts68_fit_all_aflw2000, null_index_aflw2000 = cal_nme_with_68kp(model, root_aflw2000, flist_aflw2000)
    ana_aflw2000(calc_nme_aflw2000(pts68_fit_all_aflw2000), null_index_aflw2000)

    # Calculate NME AFLW
    root_aflw = 'data/test.data/AFLW_GT_crop'
    img_list_aflw = 'data/test.data/AFLW_GT_crop.list'
    with open(img_list_aflw) as f:
        flist_aflw = [line.rstrip() for line in f]
    pts68_fit_all_aflw, null_index_aflw = cal_nme_with_68kp(model, root_aflw, flist_aflw)
    ana_aflw(calc_nme_aflw(pts68_fit_all_aflw), null_index_aflw)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass