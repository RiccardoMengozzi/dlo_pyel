import elastica as ea
from elastica.rod.cosserat_rod import CosseratRod
from elastica.callback_functions import CallBackBaseClass

from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import extend_stepper_interface


from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

from pyel_model.action import MoveAction2D
from pyel_model.contact import RodPlaneContact


class DloSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,  # Enabled to use boundary conditions
    ea.Forcing,  # Enabled to use forcing 'GravityForces'
    ea.Connections,  # Enabled to use FixedJoint
    ea.CallBacks,  # Enabled to use callback
    ea.Damping,  # Enabled to use damping models on systems.
    ea.Contact,  # Enabled to use contact models
):
    pass


class DloCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            # Save time, step number, position, orientation and velocity
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            return


@dataclass
class DloModelParams:
    dt: float
    n_elem: int
    length: float
    radius: float
    density: float
    youngs_modulus: float
    nu: float
    action_velocity: float
    poission_ratio: float = 0.25
    plane_spring_constant: float = 1e2
    plane_damping_constant: float = 1e-1
    plane_slip_velocity_tol: float = 1e-4
    plane_kinetic_mu_array: np.ndarray = np.array([3.0, 3.0, 3.0])


class DloModel:

    def __init__(self, dlo_params, position=None, directors=None):

        self.dlo_params = dlo_params
        self.dt = dlo_params.dt
        self.action_vel = dlo_params.action_velocity

        self.action = None
        self.dlo_params_dict = {
            "dt": dlo_params.dt,
            "n_elem": dlo_params.n_elem,
            "length": dlo_params.length,
            "radius": dlo_params.radius,
            "density": dlo_params.density,
            "youngs_modulus": dlo_params.youngs_modulus,
            "action_velocity": dlo_params.action_velocity,
        }

        #######################
        # constants
        self.gravity = -9.80665

        # shear modulus
        self.shear_modulus = self.dlo_params.youngs_modulus / (2 * (1 + dlo_params.poission_ratio))

        self.simulator = DloSimulator()

        # Create rod
        self.rod = CosseratRod.straight_rod(
            n_elements=dlo_params.n_elem,
            start=np.array([0.0, 0.0, dlo_params.radius]),
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            base_length=dlo_params.length,
            base_radius=dlo_params.radius,
            density=dlo_params.density,
            youngs_modulus=dlo_params.youngs_modulus,
            shear_modulus=self.shear_modulus,
        )

        if position is not None and directors is not None:
            self.rod.position_collection[:] = position[:]
            self.rod.director_collection[:] = directors[:]

            self.rod.rest_kappa[:] = self.rod.kappa[:]
            self.rod.rest_sigma[:] = self.rod.sigma[:]

        # set bending stiffness
        self.set_bending_stiffness(young_modulus=dlo_params.youngs_modulus, shear_modulus=self.shear_modulus)
        self.set_shear_stiffness_inextensible(S=10e3)

        self.simulator.append(self.rod)

        # Hold data from callback function
        self.callback_data = ea.defaultdict(list)
        self.simulator.collect_diagnostics(self.rod).using(
            DloCallBack, step_skip=1000, callback_params=self.callback_data
        )

        # integration scheme
        self.timestepper = PositionVerlet()

    def set_bending_stiffness(self, young_modulus, shear_modulus):
        I_1 = I_2 = np.pi / 4 * self.dlo_params.radius**4
        I_3 = np.pi / 2 * self.dlo_params.radius**4
        self.rod.bend_matrix[0, 0, :] = I_1 * young_modulus
        self.rod.bend_matrix[1, 1, :] = I_2 * young_modulus
        self.rod.bend_matrix[2, 2, :] = I_3 * shear_modulus

    def set_shear_stiffness_inextensible(self, S=10e5):
        self.rod.shear_matrix[0, 0, :] = S
        self.rod.shear_matrix[1, 1, :] = S
        self.rod.shear_matrix[2, 2, :] = S

    def get_callback_data(self):
        return self.callback_data

    def add_move_action(self):
        self.simulator.add_forcing_to(self.rod).using(
            MoveAction2D, action=self.action, dt=self.dt, velocity=self.action_vel
        )

    def add_plane(self, plane_normal, plane_origin):

        ground_plane = ea.Plane(plane_normal=plane_normal, plane_origin=plane_origin)
        self.simulator.append(ground_plane)

        self.simulator.detect_contact_between(self.rod, ground_plane).using(
            RodPlaneContact,
            k=self.dlo_params.plane_spring_constant,
            nu=self.dlo_params.plane_damping_constant,
            slip_velocity_tol=self.dlo_params.plane_slip_velocity_tol,
            kinetic_mu_array=self.dlo_params.plane_kinetic_mu_array,
        )

    def build_model(self, action=None):
        """
        action [idx, disp along rod axis, disp perpendicular to rod axis, rotation plane]
        """

        if action is not None:
            self.action = action
            self.add_move_action()

        # gravity
        self.simulator.add_forcing_to(self.rod).using(ea.GravityForces, acc_gravity=np.array([0.0, 0.0, self.gravity]))

        # damping
        self.simulator.dampen(self.rod).using(
            ea.AnalyticalLinearDamper, damping_constant=self.dlo_params.nu, time_step=self.dt
        )

        # plane
        self.add_plane(plane_normal=np.array([0.0, 0.0, 1.0]), plane_origin=np.array([0.0, 0.0, 0.0]))

        # finalize
        self.simulator.finalize()

    def compute_steps_from_action(self):

        tot_steps = 50000

        if not hasattr(self, "action") or self.action is None:
            return tot_steps

        max_disp_norm = np.linalg.norm(np.array(self.action[1:3]))
        return int(max_disp_norm / (self.action_vel * self.dt))

    def run_simulation(self, progress_bar=True):

        total_steps = self.compute_steps_from_action()
        # print("Total steps: ", total_steps)

        do_step, stages_and_updates = extend_stepper_interface(self.timestepper, self.simulator)

        init_shape = self.rod.position_collection.copy()
        init_directors = self.rod.director_collection.copy()

        time = 0
        if progress_bar:
            for i in tqdm(range(total_steps)):
                time = do_step(self.timestepper, stages_and_updates, self.simulator, time, self.dt)
        else:
            for i in range(total_steps):
                time = do_step(self.timestepper, stages_and_updates, self.simulator, time, self.dt)

        final_shape = self.rod.position_collection.copy()
        final_directors = self.rod.director_collection.copy()

        observation = np.array(self.callback_data["position"])

        return {
            "dlo_params": dict(self.dlo_params_dict),
            "total_steps": int(total_steps),
            "action": np.array(self.action),
            "init_shape": np.array(init_shape),
            "init_directors": np.array(init_directors),
            "final_shape": np.array(final_shape),
            "final_directors": np.array(final_directors),
            "observation": np.array(observation),
        }
