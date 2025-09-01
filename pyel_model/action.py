import elastica as ea
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
import matplotlib.pyplot as plt
from pipy.utils import rpy_to_rot, rot_to_rpy

from pyel_model.plot import plot_frame


def convert_action_to_global_frame(action, dlo_shape, dlo_directors):
    """
    action contains the following fields:
    - idx: the index of the node where the action is applied (idx, idx+1)
    - x, y, z: the displacement of the node in the local frame
    - roll, pitch, yaw: the rotation of the node in the local frame
    dlo_shape is the shape of the rod in the global frame
    dlo_directors is the orientations of the rod in the global frame
    """

    # node orientation wrt the global frame
    Q = dlo_directors[..., action[0]].T

    # node position wrt the global frame
    x_world = dlo_shape[..., action[0]]

    # node position wrt the local frame
    x = Q.T @ x_world

    # add the displacements
    new_frame_world = np.matmul(Q, rpy_to_rot([0.0, action[3], 0.0]))
    new_x = Q @ (x + np.array([action[2], 0.0, action[1]]))

    return new_x, new_frame_world


def convert_action_to_local_frame(action, dlo_shape, dlo_directors):

    action_idx = int(action[0])
    target_pos = np.array([action[2], 0.0, action[1]])  # x, y, z
    target_rot = rpy_to_rot([0.0, action[3], 0.0])  # roll, pitch, yaw

    Q = dlo_directors[..., action_idx].T  # node orientation wrt the global frame
    x_world = dlo_shape[..., action_idx]  # node position wrt the global frame

    disp_local = Q.T @ (target_pos - x_world)

    rot = Q.T @ target_rot
    rpy = rot_to_rpy(rot)

    if True:
        print("x_world", x_world)
        print("target_pos", target_pos)

        x = Q.T @ x_world
        new_frame_world = np.matmul(Q, rpy_to_rot(rpy))
        new_x = Q @ (x + np.array(disp_local))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(dlo_shape[0], dlo_shape[1], dlo_shape[2], c="b")
        plot_frame(ax, x_world, Q)
        plot_frame(ax, target_pos, target_rot)
        plot_frame(ax, new_x, new_frame_world)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    return action_idx, disp_local[0], disp_local[1], rpy[2]


class MoveAction2D(ea.NoForces):

    def __init__(self, action, dt, velocity):
        """
        action: list of action values [idx grasp, disp along rod axis, disp perpendicular to rod axix, rotation plane]
        dt: float, time step
        velocity: float, m/s
        """

        self.action_idx = int(action[0])
        self.action_disp = np.array([action[2], 0.0, action[1]])
        self.action_theta = np.array([0.0, action[3], 0.0])

        #
        self.dt = dt
        self.vel = velocity
        self.init_pos_0 = None
        self.init_pos_1 = None
        self.disp0_inc = None
        self.disp1_inc = None

        ##########################################
        disp_norm = np.linalg.norm(self.action_disp)
        if disp_norm == 0:
            self.steps = 100000
        else:
            self.steps = disp_norm / (self.vel * self.dt)

    def apply_forces(self, system, time: float = 0.0):
        curr_step = int(time / self.dt)
        if curr_step == 0:
            self.init_pos_0 = system.position_collection[..., self.action_idx].copy()
            self.init_dir_0 = system.director_collection[..., self.action_idx].copy()
            self.init_pos_1 = system.position_collection[..., self.action_idx + 1].copy()
            self.disp0_inc, self.disp1_inc = self.compute_edge_disps_increments()

            self.rpy_inc = self.action_theta / self.steps

        # Move the position
        system.position_collection[..., self.action_idx] = self.init_pos_0 + self.disp0_inc * curr_step
        system.position_collection[..., self.action_idx + 1] = self.init_pos_1 + self.disp1_inc * curr_step

        # Move the director
        rot_step = rpy_to_rot(self.rpy_inc * curr_step)
        system.director_collection[..., self.action_idx] = np.dot(self.init_dir_0.T, rot_step).T

    def compute_edge_disps_increments(self):

        edge_pos_world = (self.init_pos_0 + self.init_pos_1) / 2
        edge_dir_world = self.init_pos_1 - self.init_pos_0
        edge_len = np.linalg.norm(edge_dir_world)
        edge_dir_world = edge_dir_world / edge_len

        ##############################
        # LOCAL
        ##############################
        rot = self.init_dir_0.T
        edge_pos_local = rot.T @ edge_pos_world
        edge_dir_local = rot.T @ edge_dir_world

        # new pos
        new_edge_pos_local = edge_pos_local + self.action_disp

        # new edge dir
        new_edge_dir_local = np.dot(rpy_to_rot(self.action_theta), edge_dir_local)

        ##############################
        # WORLD
        ##############################
        new_edge_pos = rot @ new_edge_pos_local
        new_edge_dir = rot @ new_edge_dir_local

        # new pos node 0
        pos0_tgt = new_edge_pos - new_edge_dir * edge_len / 2

        # new pos node 1
        pos1_tgt = new_edge_pos + new_edge_dir * edge_len / 2

        # displacement (total)
        disp0 = pos0_tgt - self.init_pos_0
        disp1 = pos1_tgt - self.init_pos_1

        # displacement (increment)
        disp0_inc = disp0 / self.steps
        disp1_inc = disp1 / self.steps

        return disp0_inc, disp1_inc
