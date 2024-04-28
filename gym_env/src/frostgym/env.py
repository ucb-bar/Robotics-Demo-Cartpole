import sys
import os
import math
import time
from typing import Sequence

import numpy as np
import gym
from isaacgym import gymapi, gymtorch, gymutil
import torch

from .agent import PolicyGradientAgent
from .asset import GenericAsset


class InitState:
    def __init__(self,
            root_pos=gymapi.Vec3(0, 0, 1),
            root_rot=gymapi.Quat(0, 0, 0, 1),
            dof_pos=np.zeros(0),
            root_vel=gymapi.Vec3(0, 0, 0),
            root_ang_vel=gymapi.Vec3(0, 0, 0),
            dof_vel=np.zeros(0)
            ):
        self.root_pos = root_pos
        self.root_rot = root_rot
        self.dof_pos = dof_pos
        self.root_vel = root_vel
        self.root_ang_vel = root_ang_vel
        self.dof_vel = dof_vel


class GymBaseEnvironment:
    def __init__(self, env_name: str, render_mode="rgb_array"):
        self.render_mode = render_mode

        self.env = gym.make(env_name, render_mode=self.render_mode)

        self.n_obs = self.env.observation_space.shape[0]
        self.n_acs = self.env.action_space.shape[0]
    
    def reset(self) -> torch.Tensor:
        obs, info = self.env.reset()
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, acs: torch.Tensor) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(acs.to("cpu").detach().numpy())
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(terminated),
            torch.tensor(truncated),
            info)

    def sampleTrajectory(self, policy: PolicyGradientAgent, max_path_length: int) -> dict:
        obs_traj = []
        next_obs_traj = []
        acs_traj = []
        rewards_traj = []
        terminated_traj = []

        obs = self.reset()

        rollout_done = False
        steps = 0
        while not rollout_done:
            obs_batch = obs.unsqueeze(dim=0)
            acs_batch = policy.getAction(obs_batch)
            acs = acs_batch[0]

            next_obs, reward, terminated, truncated, info = self.step(acs)
            steps += 1

            rollout_done = terminated or truncated or steps >= max_path_length

            obs_traj.append(obs)
            next_obs_traj.append(next_obs)
            acs_traj.append(acs)
            rewards_traj.append(reward)
            terminated_traj.append(torch.tensor(rollout_done))

            obs = next_obs

        traj = {
            "obs": torch.stack(obs_traj),
            "next_obs": torch.stack(next_obs_traj),
            "acs": torch.stack(acs_traj),
            "rewards": torch.stack(rewards_traj),
            "terminated": torch.stack(terminated_traj)
        }

        return traj

    def sampleTrajectories(self, policy: PolicyGradientAgent, min_timesteps_per_batch: int, max_path_length: int) -> tuple:
        trajectories = []
        timesteps_this_batch = 0
        while timesteps_this_batch < min_timesteps_per_batch:
            traj = self.sampleTrajectory(policy, max_path_length)
            trajectories.append(traj)

            timesteps_this_batch += len(traj["obs"])

        return trajectories, timesteps_this_batch

    def sampleNTrajectories(self, policy: torch.nn.Module, n_trajectories: int, max_path_length: int) -> tuple:
        trajectories = []
        for _ in range(n_trajectories):
            trajectory = self.sampleTrajectory(policy, max_path_length)
            trajectories.append(trajectory)
        
        return trajectories

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def computeMetrics(self, writer, train_trajectories: Sequence[dict], eval_trajectories: Sequence[dict]) -> dict:
        train_returns = [traj["rewards"].sum().item() for traj in train_trajectories]
        eval_returns = [traj["rewards"].sum().item() for traj in eval_trajectories]

        train_episode_lengths = [len(traj["rewards"]) for traj in train_trajectories]
        eval_episode_lengths = [len(traj["rewards"]) for traj in eval_trajectories]

        metrics = {
            "train_returns": train_returns,
            "eval_returns": eval_returns,
            "train_episode_lengths": train_episode_lengths,
            "eval_episode_lengths": eval_episode_lengths
        }

        return metrics



class BaseIsaacGymConfig:
    num_envs            : int
    num_obs             : int
    num_privileged_obs  : int
    num_actions         : int

    control_decimation  : int

    asset               : GenericAsset

    physics_engine      : gymapi.SimType

    sim_params          : gymapi.SimParams

    def __init__(self):
        # number of environments
        self.num_envs = 1

        self.num_obs = 2
        self.num_privileged_obs = None
        self.num_actions = 2

        # control decimation
        self.control_decimation = 1

        # asset
        self.asset: GenericAsset = None

        # physics engine
        self.physics_engine = gymapi.SIM_PHYSX

        # simulation parameters
        self.sim_params = gymapi.SimParams()
        # Simulation step size
        self.sim_params.dt = 1.0 / 60.0
        # 3-Dimension vector representing gravity force in Newtons.
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # default Z-up
        self.sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        self.sim_params.use_gpu_pipeline = False

        if self.physics_engine == gymapi.SIM_FLEX:
            pass
        elif self.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 6
            self.sim_params.physx.num_velocity_iterations = 0
            self.sim_params.physx.num_threads = 0
            self.sim_params.physx.use_gpu = False

        





class BaseIsaacGymEnvironment:
    def __init__(self, 
                 config: BaseIsaacGymConfig,
                 compute_device="cuda:0",
                 graphics_device="cpu"):
        """
        Initializes the Isaac Gym environment.
        
        Args:
            config (IsaacGymConfig): Configuration object for the Isaac Gym environment.
            compute_device (str): Device to run the simulation on. Default is "cuda:0".
            graphics_device (str): Device to run the graphics on. Default is None, which is headless mode.
        """
        self.config = config
        self.compute_device = compute_device
        self.graphics_device = graphics_device
        self.viewer = None


        # initialize gym
        self.gym = gymapi.acquire_gym()


        # determine compute and graphics devices
        self.compute_device_type, self.compute_device_id = gymutil.parse_device_str(self.compute_device)

        if self.graphics_device is not None:
            self.graphics_device_type, self.graphics_device_id = gymutil.parse_device_str(self.graphics_device)
        else:
            self.graphics_device_type = "none"
            self.graphics_device_id = -1
        
        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if self.compute_device_type == "cuda" and self.config.sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"
        
        # copy the common parameters as environment attributes
        self.dt = self.config.sim_params.dt
        self.num_envs = self.config.num_envs
        self.num_obs = self.config.num_obs
        self.num_privileged_obs = self.config.num_privileged_obs
        self.num_actions = self.config.num_actions

        # optimization flags for PyTorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)


        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float32)
        self.reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float32)
        else:
            self.privileged_obs_buf = None



        self.enable_viewer_sync = True


        # environment handles
        self.envs = []
        self.actors = []


        # configure sim
        self.sim = self.gym.create_sim(self.compute_device_id, self.graphics_device_id, self.config.physics_engine, self.config.sim_params)

        if self.sim is None:
            print("***[FATAL] Failed to create sim")
            quit()

        # initialize simulation contexts
        self.createSim()

        # load asset
        self.loadAsset()


        self.gym.prepare_sim(self.sim)

        self.initBuffers()


        # viewer interface
        if self.graphics_device is not None:
            self.viewer = self.createViewer()
        

    
    def initBuffers(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        

    def createViewer(self):
        # create viewer
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if viewer is None:
            print("***[FATAL] Failed to create viewer")
            quit()

        # position the camera
        cam_pos = gymapi.Vec3(0, 2, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0.8)
        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        # subscribe to keyboard events
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        return viewer
    
    def createSim(self):
        """
        Set up the simulation with configured parameters.

        Returns:
            gymapi.Sim: Simulation object
        """
        pass

        # add ground plane
        self.createGroundPlane()


    def createGroundPlane(self):
        """
        Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.static_friction = self.cfg.terrain.static_friction
        # plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        # plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)


    def loadAsset(self):
        asset = self.config.asset
        asset_path = os.path.join(asset.root_path, asset.filename)
        print("Loading asset \"%s\"..." % asset_path)

        asset_handle = self.gym.load_asset(self.sim, asset.root_path, asset.filename, asset.options)

        num_bodies = self.gym.get_asset_rigid_body_count(asset_handle)
        num_dofs = self.gym.get_asset_dof_count(asset_handle)
        dof_props = self.gym.get_asset_dof_properties(asset_handle)
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset_handle)

        self.num_dofs = num_dofs

        body_names = self.gym.get_asset_rigid_body_names(asset_handle)
        dof_names = self.gym.get_asset_dof_names(asset_handle)

        assert num_bodies == len(body_names), "Body name count mismatch"
        assert num_dofs == len(dof_names), "DOF name count mismatch"

        print(" - Degrees of freedom: %d" % num_dofs)

        self.init_state = InitState(
            dof_pos=np.zeros(num_dofs),
            dof_vel=np.zeros(num_dofs)
        )


        # set up the env grid
        self.env_spacing = 2.5

        self._setEnvOrigins()

        unbounded_env = False

        if unbounded_env:
            env_lower = gymapi.Vec3(0., 0., 0.)
            env_upper = gymapi.Vec3(0., 0., 0.)
        else:
            env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
            env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)


        print("Creating %d environments" % self.num_envs)
        for i in range(self.num_envs):
            num_per_row = int(np.sqrt(self.num_envs))

            # create env
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_handle)

            # add actor
            self.pose = gymapi.Transform()
            self.pose.p = self.init_state.root_pos
            self.pose.r = self.init_state.root_rot

            # 1 to disable, 0 to enable...bitwise filter
            self_collisions = 1
            actor_handle = self.gym.create_actor(env_handle, asset_handle, self.pose, asset.name, i, self_collisions, 0)
            self.actors.append(actor_handle)

            # set default DOF positions
            dof_state = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
            dof_state["pos"] = self.init_state.dof_pos
            dof_state["vel"] = self.init_state.dof_vel

            props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
            # set to position control mode
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(100.0)
            props["damping"].fill(20.0)

            # set to torque control mode
            # props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, props)

            self.gym.set_actor_dof_states(env_handle, actor_handle, dof_state, gymapi.STATE_ALL)

        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))


    def _setEnvOrigins(self):
        """ 
        Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        pass



    def step(self, actions: torch.Tensor):
        """
        Apply actions and step the simulation.

        Args:
            actions (np.ndarray): Actions to apply to the simulation
        """
        
        clip_actions = float("inf")
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.render()
        
        for _ in range(self.config.control_decimation):
            # torques = self._compute_torques(actions)
            torques = actions
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)
            if self.compute_device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        # self.post_physics_step()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        obs = torch.cat([
            self.base_quat,
            self.dof_pos,
        ], dim=-1)




        clip_observations = float("inf")
        obs = torch.clip(obs, -clip_observations, clip_observations)



        #     # self.gym.set_dof_position_target_tensor_indexed(self.envs[i], )
        #     self.gym.set_actor_dof_states(self.envs[i], self.actors[i], actions_gym, gymapi.STATE_POS)
        #     self.gym.set_rigid_transform(self.envs[i], 0, self.pose)

            # self.gym.set_actor_dof_position_targets(self.envs[i], self.actors[i], position_targets)

            # torque mode
            # self.gym.set_actor_dof_states(self.envs[i], self.actors[i], actions_gym, gymapi.STATE_EFFORT)

        self.gym.simulate(self.sim)
        self.render(False)


    def _reset_dofs(self, env_ids):
        """ 
        Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """
        Resets ROOT states position and velocities of selected environmments
        Sets base position based on the curriculum
        Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    
    def render(self, sync_frame_time=True):
        if not self.viewer:
            return
        
        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
        
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # fetch results
        if self.compute_device != "cpu":
            self.gym.fetch_results(self.sim, True)
        
        if self.enable_viewer_sync:
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            if sync_frame_time:
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)


    def stop(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


    def run(self):
        # get array of DOF properties
        dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)

        dof_props = self.gym.get_asset_dof_properties(asset_handle)

        # get list of DOF types
        dof_types = [self.gym.get_asset_dof_type(asset_handle, i) for i in range(self.num_dofs)]


        # get the limit-related slices of the DOF properties array
        stiffnesses = dof_props['stiffness']
        dampings = dof_props['damping']
        armatures = dof_props['armature']
        has_limits = dof_props['hasLimits']
        self.lower_limits = dof_props['lower']
        self.upper_limits = dof_props['upper']

        # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        self.default_dof_pos = np.zeros(self.num_dofs)
        self.speeds = np.zeros(self.num_dofs)
        speed_scale = 1

        for i in range(self.num_dofs):
            if has_limits[i]:
                if dof_types[i] == gymapi.DOF_ROTATION:
                    self.lower_limits[i] = torch.clip(self.lower_limits[i], -math.pi, math.pi)
                    self.upper_limits[i] = torch.clip(self.upper_limits[i], -math.pi, math.pi)
                # make sure our default position is in range
                if self.lower_limits[i] > 0.0:
                    self.default_dof_pos[i] = self.lower_limits[i]
                elif self.upper_limits[i] < 0.0:
                    self.default_dof_pos[i] = self.upper_limits[i]
            else:
                # set reasonable animation limits for unlimited joints
                if dof_types[i] == gymapi.DOF_ROTATION:
                    # unlimited revolute joint
                    self.lower_limits[i] = -math.pi
                    self.upper_limits[i] = math.pi
                elif dof_types[i] == gymapi.DOF_TRANSLATION:
                    # unlimited prismatic joint
                    self.lower_limits[i] = -1.0
                    self.upper_limits[i] = 1.0
            # set DOF position to default
            dof_states["pos"][i] = self.default_dof_pos[i]
            # set speed depending on DOF type and range of motion
            if dof_types[i] == gymapi.DOF_ROTATION:
                self.speeds[i] = speed_scale * np.clip(2 * (self.upper_limits[i] - self.lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
            else:
                self.speeds[i] = speed_scale * np.clip(2 * (self.upper_limits[i] - self.lower_limits[i]), 0.1, 7.0)

        # Print DOF properties
        for i in range(self.num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % self.dof_names[i])
            print("  Type:     %s" % self.gym.get_dof_type_string(dof_types[i]))
            print("  Stiffness:  %r" % stiffnesses[i])
            print("  Damping:  %r" % dampings[i])
            print("  Armature:  %r" % armatures[i])
            print("  Limited?  %r" % has_limits[i])
            if has_limits[i]:
                print("    Lower   %f" % self.lower_limits[i])
                print("    Upper   %f" % self.upper_limits[i])


        # joint animation states
        ANIM_SEEK_LOWER = 1
        ANIM_SEEK_UPPER = 2
        ANIM_SEEK_DEFAULT = 3
        ANIM_FINISHED = 4

        # initialize animation state
        anim_state = ANIM_SEEK_LOWER
        current_dof = 0
        print("Animating DOF %d ('%s')" % (current_dof, self.dof_names[current_dof]))

        # ref_motion = np.load(ref_motion, allow_pickle=True).tolist()
        # ref_dof_pos = ref_motion["dof_pos"]
        ref_motion = None
        
        dof_positions = np.zeros(self.num_dofs)
        
        show_axis = False

        while True:
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not ref_motion:

                speed = self.speeds[current_dof]

                # animate the dofs
                if anim_state == ANIM_SEEK_LOWER:
                    dof_positions[current_dof] -= speed * self.dt
                    if dof_positions[current_dof] <= self.lower_limits[current_dof]:
                        dof_positions[current_dof] = self.lower_limits[current_dof]
                        anim_state = ANIM_SEEK_UPPER
                elif anim_state == ANIM_SEEK_UPPER:
                    dof_positions[current_dof] += speed * self.dt
                    if dof_positions[current_dof] >= self.upper_limits[current_dof]:
                        dof_positions[current_dof] = self.upper_limits[current_dof]
                        anim_state = ANIM_SEEK_DEFAULT
                if anim_state == ANIM_SEEK_DEFAULT:
                    dof_positions[current_dof] -= speed * self.dt
                    if dof_positions[current_dof] <= self.default_dof_pos[current_dof]:
                        dof_positions[current_dof] = self.default_dof_pos[current_dof]
                        anim_state = ANIM_FINISHED
                elif anim_state == ANIM_FINISHED:
                    dof_positions[current_dof] = self.default_dof_pos[current_dof]
                    current_dof = (current_dof + 1) % self.num_dofs
                    anim_state = ANIM_SEEK_LOWER
                    print("Animating DOF %d ('%s')" % (current_dof, self.dof_names[current_dof]))


            else:
                for i in range(24):
                    dof_positions[i] = ref_dof_pos[frame_idx, i]

                self.pose.p = gymapi.Vec3(ref_motion["root_pos"][frame_idx, 0], ref_motion["root_pos"][frame_idx, 1], ref_motion["root_pos"][frame_idx, 2])
                self.pose.r = gymapi.Quat(
                    ref_motion["root_rot"][frame_idx, 1], 
                    ref_motion["root_rot"][frame_idx, 2], 
                    ref_motion["root_rot"][frame_idx, 3],
                    ref_motion["root_rot"][frame_idx, 0]
                )


                # flip direction in Isaac Gym
                dof_positions *= -1

                frame_idx += 1
                if frame_idx >= ref_motion['root_pos'].shape[0]:
                    frame_idx = 0

            if show_axis:
                self.gym.clear_lines(self.viewer)

            for i in range(self.num_envs):
                if show_axis:
                    # get the DOF frame (origin and axis)
                    dof_handle = self.gym.get_actor_dof_handle(self.envs[i], self.actors[i], current_dof)
                    frame = self.gym.get_dof_frame(self.envs[i], dof_handle)

                    # draw a line from DOF origin along the DOF axis
                    p1 = frame.origin
                    p2 = frame.origin + frame.axis * 0.7
                    color = gymapi.Vec3(1.0, 0.0, 0.0)
                    gymutil.draw_line(p1, p2, color, self.gym, self.viewer, self.envs[i])

            self.step(torch.Tensor(dof_positions))
            time.sleep(0.01)

        print("Done")
        self.stop()
