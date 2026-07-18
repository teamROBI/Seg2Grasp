import time

def induction_basic(agent, vision, config, op_time=600):
    home_pos = config['pos_home']
    place_pose = config['pos_place']
    box_height = config['box_height']
    start_time = time.time()
    # agent.speedj(GLOBAL_VEL, GLOBAL_ACC, 180)
    vel = [agent.vel for x in range(6)]
    acc = agent.acc
    agent.robot.speedj(vel, acc, op_time)
    while (time.time() - start_time) < op_time:
        print('home_pos', home_pos)
        agent.movej(home_pos)
        print('1. go home')
        pc, obj_name, bbx = vision.get_object_pc()
        print("pc: ", pc)
        print("vision.pc: ", vision.pc)

        if pc is None:
            continue
        dist_x, dist_y, dist_z = agent.calculate_dist_for_pick(pc)
        if not agent.check_valid_reach_in_home_pos(dist_x, dist_y, dist_z):
            print('[SYS] invalid reach point')
            continue
        # 2. go for pick
        print('2. pick', obj_name, 'go for pick', (round(dist_x, 3), round(dist_y, 3), round(dist_z, 3)),
              'object height', round(pc[2] - box_height, 4))

        agent.movel((dist_x, dist_y, dist_z + 0.1), relative=True)

        agent.movel((0, 0, -0.1), acc=1.5, vel=1.5, relative=True)

        # 3. pick
        agent.gripper.open_gripper(sleep=1.5)
        # 4. go transit
        print('3. go transit')
        agent.movel((0, 0, 0.1), relative=True)
        agent.movej(home_pos)
        print('vaccum pos:', agent.get_vaccum_pos())
        if agent.get_vaccum_pos() == 100:
            continue
        # 5. go for place
        print('4. go for place')
        agent.movej(place_pose)
        agent.movel((0, 0, -0.1), relative=True)
        agent.gripper.close_gripper(sleep=0.1)
        agent.movej(place_pose)


def induction_with_ransac(agent, vision, config, op_time=600):
    home_pos = config['pos_home']
    place_pose = config['pos_place']
    box_height = config['box_height']
    start_time = time.time()
    vel = [agent.vel for x in range(6)]
    acc = agent.acc
    agent.robot.speedj(vel, acc, op_time)
    while (time.time() - start_time) < op_time:
        agent.movej(home_pos)
        print('1. go home')
        pc, obj_name, bbx, roll_pitch_yaw = vision.get_object_pc_with_ransac()
        if pc is None:
            continue
        roll_pitch_yaw[0] = 0
        dist_x, dist_y, dist_z = agent.calculate_dist_for_pick(pc)
        if not agent.check_valid_reach_in_home_pos(dist_x, dist_y, dist_z):
            print('[SYS] invalid reach point')
            continue
        # 2. go for pick
        print('2. pick', obj_name, 'go for pick', (round(dist_x, 3), round(dist_y, 3), round(dist_z, 3)),
              'object height', round(pc[2] - box_height, 4))
        # rotate move
        agent.rotate_movel([dist_x, dist_y, dist_z], roll_pitch_yaw)
        # continue
        # 3. pick
        agent.gripper.open_gripper(sleep=1.5)
        # 4. go transit
        print('3. go transit')
        agent.movel((0, 0, 0.1), relative=True)
        agent.movej(home_pos)
        print('vaccum pos:', agent.get_vaccum_pos())
        if agent.get_vaccum_pos() == 100:
            continue
        # 5. go for place
        print('4. go for place')
        agent.movej(place_pose)
        agent.movel((0, 0, -0.1), relative=True)
        agent.gripper.close_gripper(sleep=0.1)
        agent.movej(place_pose)