from robot_module.module.urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import numpy as np
import socket
import math
from scipy.spatial.transform import Rotation as R
import robot_module.module.utils as utils
import threading
import queue
import json
with open('robot_module/config.json', 'r') as f:
    config = json.load(f)

L_B_D = [0, -1, 0, -0.25, -0.7071, 0, -0.7071, 1.0323, 0.7071, 0, -0.7071, -0.1838, 0, 0, 0, 1]
L_B_D = np.array(L_B_D).reshape(4, 4)
ROBOT_IP = config['robot_ip']
PORT = 63352


class Agent:
    def __init__(self, robot, acc, vel):
        self.robot = robot
        self.L_B_D = L_B_D
        self.gripper = Robotiq_Two_Finger_Gripper(self.robot, speed=200, force=20)
        self.vel = vel
        self.acc = acc
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.task_queue = TaskHandler(self.robot, self.task_queue, self.result_queue)
        self.task_queue.start()

    def movel(self, tpose, acc=None, vel=None, wait=True, relative=False):
        if acc == None:
            acc = self.acc
        if vel == None:
            vel = self.vel
        if len(tpose) == 3:
            if relative:
                tpose = np.array(self.robot.getl()[:3]) + tpose
            self.robot.movel(tpose=np.concatenate([tpose, self.robot.getl()[3:]]), acc=acc, vel=vel, wait=wait, relative=False)
        elif len(tpose) == 6:
            self.robot.movel(tpose, acc=acc, vel=vel, wait=wait, relative=relative)
        else:
            assert False, 'length of tcp input must be 3 or 6'

    def movej(self, joints, acc=None, vel=None, wait=True, relative=False):
        if acc == None:
            acc = self.acc
        if vel == None:
            vel = self.vel
        self.robot.movej(joints, acc=acc, vel=vel, wait=wait, relative=relative)

    def move_rotate(self, robot_xyz, rot_info, nv, calibration_ratio=0.195, arm_len=0.20, offset=0.05):
        # arm length = 15, 20 ,24
        rv = rot_info[0]
        rpy = rot_info[1]
        rot_mat = rot_info[2]
        angle = np.arccos(np.dot([0, 0, 1], nv))
        print("rv:", rv)
        print("rpy:", rpy)
        print("normal vector:", nv)
        print("angle:", angle)

        if nv[2] > 0.985:
            self.movel((robot_xyz[0], robot_xyz[1], robot_xyz[2] + 0.1), relative=True)
            self.movel((0, 0, -0.098), acc=1.0, vel=1.0, relative=True)
        else:
            calibration_xyz = [nv[1] * calibration_ratio, nv[0] * calibration_ratio, arm_len * (1 - math.cos(angle))] # 0.014
            offset_xyz = [offset * nv[1], offset * nv[0], offset * nv[2]]
            print("cali z", calibration_xyz[2])

            trans = self.robot.get_pose()  # get current transformation matrix (tool to base)
            trans.pos.x += robot_xyz[0] + calibration_xyz[0] + offset_xyz[0]
            trans.pos.y += robot_xyz[1] + calibration_xyz[1] + offset_xyz[1]
            trans.pos.z += robot_xyz[2] - calibration_xyz[2] + offset_xyz[2]
            trans.orient = rot_mat
            self.robot.set_pose(trans, acc=self.acc, vel=self.vel)  # appclely the new pose

            z_dist = np.sqrt(offset_xyz[0]**2 + offset_xyz[1]**2 + offset_xyz[2]**2)
            print("z dist", z_dist)

            self.robot.translate_tool((0, 0, z_dist + 0.007), acc=self.acc, vel=self.vel)

    def getj(self):
        return self.robot.getj()

    def getl(self):
        return self.robot.getl()

    def get_pose(self):
        return self.robot.get_pose()

    def stopl(self, a):
        self.robot.stopl(a)

    def stopj(self, a):
        self.robot.stopj(a)

    def stop(self):
        self.robot.stop()

    def moveD(self, tcp, acc=None, vel=None, wait=True, relative=False):
        if acc == None:
            acc = self.acc
        if vel == None:
            vel = self.vel
        if len(tcp) == 3:
            if relative:
                dir = self.desk_to_base(direction=tcp)
                self.movel(dir, acc=acc, vel=vel, wait=wait, relative=True)
            else:
                pos = self.desk_to_base(position=tcp)
                self.movel(pos, acc=acc, vel=vel, wait=wait, relative=False)
        elif len(tcp) == 6:
            if relative:
                # print('0. current_tcp_D', self.getl())
                current_tcp_D = utils.affine_to_tcp(self.base_to_desk(affine=utils.tcp_to_affine(self.getl())))
                # print('1. current_tcp_D', current_tcp_D)
                tcp_D = current_tcp_D + tcp
                # print('2. tcp_D', tcp_D)
                tcp_B = utils.affine_to_tcp(self.desk_to_base(affine=utils.tcp_to_affine(tcp_D)))
                # print('3. tcp_B', tcp_B)
                self.movel(tcp_B, acc=acc, vel=vel, wait=wait, relative=False)
            else:
                tcp_B = utils.affine_to_tcp(self.desk_to_base(affine=utils.tcp_to_affine(tcp)))
                self.movel(tcp_B, acc=acc, vel=vel, wait=wait, relative=False)
        else:
            assert False, 'length of tcp input must be 3 or 6'

    def desk_to_base(self, position=None, direction=None, scipy_R=None, dcm=None, affine=None):
        return utils.transform(self.L_B_D, position, direction, scipy_R, dcm, affine)

    def base_to_desk(self, position=None, direction=None, scipy_R=None, dcm=None, affine=None):
        return utils.transform(np.linalg.inv(self.L_B_D), position, direction, scipy_R, dcm, affine)

    def convert_cam2robot_coord_for_pick(self, pc, offset=0.1):
        """
        :param pc: point cloud
        :param nv: normal vector
        :return:
        """
        cam_to_robot_diff = [-0.025, -0.292, 0.84]
        pc[0], pc[1], pc[2] = round(pc[0], 3), round(pc[1], 3), round(pc[2], 3)
        # camera side: +x = back, +y = left, +z = down
        coord_x = -pc[1] + cam_to_robot_diff[0]
        coord_y = -pc[0] + cam_to_robot_diff[1]
        coord_z = -pc[2] + cam_to_robot_diff[2]

        return coord_x, coord_y, coord_z

    def check_valid_reach_in_home_pos(self, robot_x, robot_y, robot_z):
        # print(robot_x, robot_y, robot_z)
        # [0.242, -0.447. -0.176] right top
        # [-0.24, -0.135, -0.197] left bottom
        if robot_x < -0.25 or robot_x > 0.25:
            print(f"range x not -0.25 < {robot_x} < 0.25")
            return False
        if robot_y < -0.45 or robot_y > -0.12:
            print(f"range y not -0.45 < {robot_y} < -0.12")
            return False
        if robot_z < -0.3 or robot_z > 0:
            print(f"range z not -0.3 < {robot_z} < 0")
            return False
        return True

    def get_vaccum_pos(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ROBOT_IP, PORT))
        s.send(b"socket_send_string('GET POS', 'gripper_socket')")
        data = s.recv(1024)
        return int(data)


    def rotate_movel(self, dist_xyz, rpy, offset=0.05, arm_len=0.15):
        (roll, pitch, yaw) = rpy
        print('[roll, pitch, yaw]', round(roll * 180 / math.pi),
                                    round(pitch * 180 / math.pi),
                                    round(yaw * 180 / math.pi))
        trans = self.robot.get_pose()  # get current transformation matrix (tool to base)
        trans.pos.x += dist_xyz[0] + math.tan(pitch) * arm_len
        trans.pos.y += dist_xyz[1] - math.tan(-yaw) * arm_len
        trans.pos.z += dist_xyz[2] + offset

        trans.orient.rotate_xb(-yaw)
        trans.orient.rotate_yb(pitch)
        trans.orient.rotate_zb(roll)
        self.robot.set_pose(trans, acc=self.acc, vel=self.vel)  # apply the new pose

        trans = self.robot.get_pose()
        trans.pos.z -= offset
        self.robot.set_pose(trans, acc=self.acc, vel=self.vel)  # apply the new pose

    def rotate_moveD(self, dist_xyz, rpy, offset=0.05, arm_len=0.15):
        # for x, y, z
        dist_xyz = dist_xyz + [0.0, 0.0, 0.0]
        current_tcp_D = utils.affine_to_tcp(self.base_to_desk(affine=utils.tcp_to_affine(self.getl())))
        target_tcp_D = current_tcp_D + dist_xyz
        print('target_tcp_D', target_tcp_D)
        target_affine_D = utils.tcp_to_affine(target_tcp_D)

        # for rotate
        (roll, pitch, yaw) = rpy
        print('[roll, pitch, yaw]', round(rpy[0] * 180 / math.pi),
              round(rpy[1] * 180 / math.pi),
              round(rpy[2] * 180 / math.pi))
        # east, west, south, north : -yaw, +yaw, -pitch, +pitch
        # rotate
        rot_mat = R.from_rotvec(np.array([-pitch, -yaw, -roll])).as_dcm()
        target_affine_rotated = utils.transform(utils.to_44(rot_mat), affine=target_affine_D)
        target_tcp_rotated = utils.affine_to_tcp(target_affine_rotated)
        # calculate offsets
        target_tcp_rotated[0] += math.tan(yaw) * arm_len  # x axis
        target_tcp_rotated[1] -= math.tan(pitch) * arm_len  # y axis
        target_tcp_rotated[2] += offset    # z axis

        # reverse rotate
        target_affine_reverse_rotated = utils.tcp_to_affine(target_tcp_rotated)
        reverse_rot_mat = R.from_rotvec(np.array([pitch, yaw, roll])).as_dcm()
        target_affine_D = utils.transform(utils.to_44(reverse_rot_mat), affine=target_affine_reverse_rotated)

        # only for rx, ry, rz
        target_rot_D = utils.transform(reverse_rot_mat, dcm=target_affine_D)
        # dcm = rotated by reverse_rot_mat, tcp = by target_affine_D (reversed desk affine matrix)
        rot_desk = utils.affine_to_tcp(dcm=target_rot_D, t=utils.affine_to_tcp(target_affine_D)[:3])
        rot_base = utils.affine_to_tcp(self.desk_to_base(affine=utils.tcp_to_affine(rot_desk)))

        self.movel(rot_base, acc=self.acc, vel=self.vel, wait=True, relative=False)

        ## for picking
        target_tcp_rotated[2] -=offset
        target_affine_reverse_rotated = utils.tcp_to_affine(target_tcp_rotated)
        target_affine_D = utils.transform(utils.to_44(reverse_rot_mat), affine=target_affine_reverse_rotated)

        rot_desk = utils.affine_to_tcp(dcm=target_rot_D, t=utils.affine_to_tcp(target_affine_D)[:3])
        rot_base = utils.affine_to_tcp(self.desk_to_base(affine=utils.tcp_to_affine(rot_desk)))

        self.movel(rot_base, acc=self.acc, vel=self.vel, wait=True, relative=False)


class TaskHandler(threading.Thread) :
    def __init__(self, robot, task_queue, result_queue):
        threading.Thread.__init__(self)
        self.robot = robot
        self.task_queue = task_queue
        self.result_queue = result_queue
    def add_task(self, task, **kwargs):
        self.task_queue.put([task, kwargs])


    def run(self):
        while True:
            task, kwargs = self.task_queue.get()
            print(kwargs)
            if task is None:
                break
            result = task(**kwargs['kwargs'])
            self.result_queue.put(result)
            self.task_queue.task_done()

