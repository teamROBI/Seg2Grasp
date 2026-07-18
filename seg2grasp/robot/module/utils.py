import numpy as np
from scipy.spatial.transform import Rotation as R
import subprocess
import signal
import traceback


def convert_ui_to_py() :
    UI_PATH = "./gui/pyside2_uis/"
    #ret = subprocess.Popen('bash {} pyside2-uic {}contoller.ui -o {}controller_ui.py'.format(VIRTUALENV_PATH,UI_PATH,UI_PATH))
    ret = subprocess.Popen('pyside2-uic {}/controller.ui -o {}/controller_ui.py'.format(UI_PATH,UI_PATH), shell=True)#.format(VIRTUALENV_PATH,UI_PATH,UI_PATH))
    print("Converting .ui to .py finished")

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
      parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
      return

    children = parent.children(recursive=True)
    for process in children:
      process.send_signal(sig)



# Converts a 3x3 matrix into corresponding 4x4 affine matrix, with given translation vector
def to_44(mat_33, t = [0, 0, 0]):
    return np.concatenate(
                [np.concatenate(
                    [mat_33, [[0, 0, 0]]], 0
                ), [[t[0]], [t[1]], [t[2]], [1]]], 1
            )

# Converts a 4x4 matrix into corresponding 3x3 matrix
def to_33(mat_44):
    return mat_44[:3, :3]

# Converts a dope output from DopeReader into corresponding affine matrix
def dope_to_affine(dope, scale=1):
    return to_44(R.from_quat(dope[3:7]).as_dcm(), np.array(dope[:3]) * scale)

# Converts an ur5 tcp(6-dimensional) into corresponding affine matrix(L_B_W)
def tcp_to_affine(tcp):
    affine = R.from_rotvec(tcp[3:]).as_dcm()
    return to_44(affine, tcp[:3])

# Converts an affine matrix(L_B_W) into corresponding ur5 tcp
def affine_to_tcp(affine=None, dcm=None, t=None):
    if affine is not None:
        pos, rot = affine[:3, 3], affine[:3, :3]
    if dcm is not None and t is not None:
        pos, rot = t, dcm

    return np.concatenate([pos, R.from_dcm(rot).as_rotvec()])

# T : 4x4 matrix, or scipy.spatial.transform.Rotation
# position : 3d vector
# direction : 3d vector
# scipy_R : scipy.spatial.transform.Rotation
# dcm : 3x3 matrix
# affine : 4x4 matrix
# Applies transform T to a given position, direction, rotation or affine transformation
def transform(T, position=None, direction=None, scipy_R=None, dcm=None, affine=None):
    if isinstance(T, R):
        T = to_44(T.as_dcm())

    if position is not None:
        p = [position[0], position[1], position[2], 1]
        return np.matmul(T, p)[:3]
    if direction is not None:
        return np.matmul(T[:3, :3], direction)
    if scipy_R is not None:
        return np.matmul(T[:3, :3], scipy_R.as_dcm())
    if dcm is not None:
        return np.matmul(T[:3, :3], dcm[:3, :3])
    if affine is not None:
        return np.matmul(T, affine)

import signal
import psutil
## jskang added
def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    print("killproc")
    try:
      parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
      return
    children = parent.children(recursive=True)
    print(children)
    for process in children:
      process.send_signal(sig)


# Exception handling
def exception_handler(exception):
    func_name = traceback.extract_tb(exception.__traceback__)[-1][2]
    print("Exception in {} : {}".format(func_name, exception))
    print(traceback.format_exc())


def check_and_modify_image_format_for_display(img):
    img_dim = img.shape
    c_idx = False
    if len(img_dim) == 2 :
        assert True, "image channel index not found"
    else :
        for i in range(len(img_dim)) :
            if img_dim[i] == 3 :
                c_idx = i
    assert c_idx != False, "image channel index not found"
    # arrange image array to (height, width, channel)
    if c_idx == 0 :
        img = np.transpose(img, (1,2,0))
    elif c_idx == 1 :
        img = np.transpose(img, (2,0,1))
    elif c_idx == 2 :
        img = np.transpose(img, (0,1,2))
    return img