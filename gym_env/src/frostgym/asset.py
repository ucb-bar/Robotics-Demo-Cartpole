import sys
import os
import math
import time

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch


class GenericAsset:
    # name of actor used in the simulation
    name = "generic_asset"

    root_path = "assets/"
    filename = "urdf/asset.urdf"

    options = gymapi.AssetOptions()

    def __init__(self):
        # Angular velocity damping for rigid bodies. Default is 0.5.
        self.options.angular_damping = 0.
        
        # The value added to the diagonal elements of inertia tensors for all of the asset’s rigid
        # bodies/links. Could improve simulation stability. Default is 0.0.
        self.options.armature = 0.
        
        # Merge links that are connected by fixed joints.
        # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        self.options.collapse_fixed_joints = True
        
        # Whether to treat submeshes in the mesh as the convex decomposition of the mesh. 
        # Default is False.
        self.options.convex_decomposition_from_submeshes = False

        # Default mode used to actuate Asset joints. See isaacgym.gymapi.DriveModeFlags.
        # (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self.options.default_dof_drive_mode = 3

        # Default density parameter used for calculating mass and inertia tensor when no mass and
        # inertia data are provided, in kg/m^3. Default is 1000.0.
        self.options.density = .001

        # Disables gravity for asset. DEfault is False.
        self.options.disable_gravity = False

        # Enable gyroscopic forces. Default is True.
        self.options.enable_gyroscopic_forces = False

        # Set Asset base to a fixed placement upon import.
        self.options.fix_base_link = True

        # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
        self.options.flip_visual_attachments = False

        # Linear velocity damping for rigid bodies. Default is 0.0.
        self.options.linear_damping = 0.

        # Maximum angular velocity for rigid bodies. In rad/s. Default is 64.0.
        self.options.max_angular_velocity = 1000.

        # Maximum linear velocity for rigid bodies. In m/s. Default is 1000.0.
        self.options.max_linear_velocity = 1000.

        # How to load normals for the meshes in the asset. One of FROM_ASSET, 
        # COMPUTE_PER_VERTEX, or COMPUTE_PER_FACE. Defaults to FROM_ASSET, falls back to 
        # COMPUTE_PER_VERTEX if normals not fully specified in mesh.
        self.options.mesh_normal_mode = gymapi.MeshNormalMode.FROM_ASSET

        # Minimum mass for particles in soft bodies, in Kg. Default is 9.999999960041972e-13
        self.options.min_particle_mass = 1e-12

        # Whether to compute the center of mass from geometry and override values given in the 
        # original asset.
        self.options.override_com = False

        # Whether to compute the inertia tensor from geometry and override values given in the
        # original asset.
        self.options.override_inertia = False

        # flag to replace Cylinders with capsules for additional performance.
        self.options.replace_cylinder_with_capsule = False

        # Number of faces on generated cylinder mesh, excluding top and bottom. Default is 20.
        self.options.slices_per_cylinder = 20

        # Default tendon limit stiffness. Choose small as the limits are not implicitly solved. 
        # Avoid oscillations by setting an apporpriate damping value. Default is 1.0.
        self.options.tendon_limit_stiffness = 1.0

        # Thickness of the collision shapes. Sets how far objects should come to rest from the 
        # surface of this body. Default is 0.019999999552965164.
        self.options.thickness = 0.01

        # Whether to use materials loaded from mesh files instead of the materials defined in
        # asset file. Default False
        self.options.use_mesh_materials = True

        # Use joint space armature instead of links inertia tensor modififcations. Default is True.
        self.options.use_physx_armature = True
        
        # Whether convex decomposition is enabled. Used only with PhysX. Default False.
        self.options.vhacd_enabled = False

        # Convex decomposition parameters. Used only with PhysX. If not specified, all triangle
        # meshes will be approximated using a single convex hull.
        self.options.vhacd_params = self.options.vhacd_params


class CartPoleAsset(GenericAsset):
    name = "generic_asset"
    filename = "urdf/cartpole.urdf"

    def __init__(self):
        super().__init__()


