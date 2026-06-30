# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from isaacsim.storage.native import get_assets_root_path
from omni.usd import StageEventType
from pxr import Sdf, UsdLux, UsdGeom, UsdPhysics, Gf, UsdShade

from .scenario import ExampleScenario


class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # Run initialization for the provided example
        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)
        
        # The section below creates a new collapsable frame with 3 float parameters

        parameters_frame = CollapsableFrame("Parameters", collapsed=False)

        with parameters_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                with ui.HStack():
                    ui.Label("Number of Segments", width=ui.Percent(40))
                    self._float_field_1 = ui.FloatField()
                    self._float_field_1.model.set_value(self._float_param_1)
                    self._float_field_1.model.add_value_changed_fn(
                        lambda m: self._on_float_param_1_changed(m.get_value_as_float())
                    )

                with ui.HStack():
                    ui.Label("Segment Length (m)", width=ui.Percent(40))
                    self._float_field_2 = ui.FloatField()
                    self._float_field_2.model.set_value(self._float_param_2)
                    self._float_field_2.model.add_value_changed_fn(
                        lambda m: self._on_float_param_2_changed(m.get_value_as_float())
                    )

                with ui.HStack():
                    ui.Label("Capsule Radius (m)", width=ui.Percent(40))
                    self._float_field_3 = ui.FloatField()
                    self._float_field_3.model.set_value(self._float_param_3)
                    self._float_field_3.model.add_value_changed_fn(
                        lambda m: self._on_float_param_3_changed(m.get_value_as_float())
                    )

        #------------------------------------------------------------------------------------

    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._scenario = ExampleScenario()
        self._cable_path = None
        
        # Initialize float parameters for cable creation
        self._float_param_1 = 30.0  # Number of segments
        self._float_param_2 = 0.02   # Height/length of each segment
        self._float_param_3 = 0.0025  # Radius of the capsule

    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_poses(np.array([[6.5, 0, 12]]))

    def _create_cable(self, base_path="/World/Cable", start_position=None, num_segments=30, segment_length=0.02, radius=0.0025):
        """
        Creates a cable made of capsule segments connected by spherical joints.
        
        Args:
            base_path: Base USD path for the cable
            start_position: Starting position [x, y, z] of the cable (default: [0, 0, 1.0])
            num_segments: Number of capsule segments
            segment_length: Length/height of each capsule segment
            radius: Radius of each capsule
        """
        if start_position is None:
            start_position = [0, 0, 1.0]
        
        stage = get_current_stage()
        num_segments = int(num_segments)
        
        # Create a parent xform for the cable
        cable_xform = UsdGeom.Xform.Define(stage, base_path)
        
        # Create folders for segments and joints
        segments_folder = UsdGeom.Xform.Define(stage, f"{base_path}/Segments")
        joints_folder = UsdGeom.Xform.Define(stage, f"{base_path}/Joints")
        
        # Enable physics on the cable parent
        UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        
        # Create black plastic material
        material_path = f"{base_path}/Materials/BlackPlastic"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.05, 0.05))  # Dark gray/black
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # Create physics material for friction properties
        physics_material_path = f"{base_path}/Materials/CablePhysicsMaterial"
        physics_material = UsdPhysics.MaterialAPI.Apply(UsdShade.Material.Define(stage, physics_material_path).GetPrim())
        physics_material.CreateStaticFrictionAttr(0.5)  # Static friction coefficient (0-1+)
        physics_material.CreateDynamicFrictionAttr(0.5)  # Dynamic/kinetic friction coefficient (0-1+)
        physics_material.CreateRestitutionAttr(0.0)  # Bounciness (0=no bounce, 1=perfect bounce)
        
        capsule_prims = []
        
        # Calculate total capsule length including hemispherical caps
        
        # Use accumulator to avoid floating-point multiplication errors
        current_z_position = start_position[2]
        
        for i in range(num_segments):
            # Create capsule path in Segments folder
            capsule_path = f"{base_path}/Segments/segment_{i}"
            
            # Create capsule geometry
            capsule = UsdGeom.Capsule.Define(stage, capsule_path)
            capsule.CreateHeightAttr(segment_length)
            capsule.CreateRadiusAttr(radius)
            capsule.CreateAxisAttr("Z")
            
            # Apply black plastic material to the capsule
            UsdShade.MaterialBindingAPI(capsule).Bind(material)
            
            # Set position (vertically stacked)
            z_offset = start_position[2] - (i * segment_length)
            capsule_xform = UsdGeom.Xformable(capsule)
            capsule_xform.ClearXformOpOrder()
            translate_op = capsule_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(start_position[0], start_position[1], z_offset))
            # position_i += segment_length
            # translate_op.Set(Gf.Vec3d(start_position[0], start_position[1], current_z_position))
            # current_z_position -= segment_length
            
            # Add physics properties
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
            
            # Apply physics material for friction
            physics_material_binding = UsdShade.MaterialBindingAPI(capsule.GetPrim())
            physics_material_binding.Bind(
                UsdShade.Material(stage.GetPrimAtPath(physics_material_path)),
                UsdShade.Tokens.weakerThanDescendants,
                "physics"
            )
            
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
            
            # First segment is fixed (attached to world)
            if i == 0:
                rigid_body_api.CreateKinematicEnabledAttr(True)
            else:
                # Set mass properties
                mass_api = UsdPhysics.MassAPI.Apply(capsule.GetPrim())
                mass_api.CreateMassAttr(0.0006)
                
                # Add damping to reduce oscillation (makes cable more realistic)
                # capsule.GetPrim().CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(6.0)
                # capsule.GetPrim().CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(12.0)
            
            capsule_prims.append(capsule_path)
            
            # Create D6 joint connecting this segment to previous one
            if i > 0:
                joint_path = f"{base_path}/Joints/joint_{i}"
                d6_joint = UsdPhysics.Joint.Define(stage, joint_path)
                
                # Set the two bodies being connected
                d6_joint.CreateBody0Rel().SetTargets([capsule_prims[i-1]])
                d6_joint.CreateBody1Rel().SetTargets([capsule_prims[i]])
                
                # Set joint local positions (at the connection point between segments)
                d6_joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, -segment_length/2))
                d6_joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, segment_length/2))

                # Apply rotation limits (degrees)
                rot_x_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "rotX")
                rot_x_limit.CreateLowAttr(-60.0)
                rot_x_limit.CreateHighAttr(60.0)

                rot_y_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "rotY")
                rot_y_limit.CreateLowAttr(-60.0)
                rot_y_limit.CreateHighAttr(60.0)

                rot_z_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "rotZ")
                rot_z_limit.CreateLowAttr(1.0)    # low > high = locked in PhysX
                rot_z_limit.CreateHighAttr(-1.0)

                # Apply translation limits (locked: low == high or low > high)
                trans_x_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "transX")
                trans_x_limit.CreateLowAttr(1.0)
                trans_x_limit.CreateHighAttr(-1.0)

                trans_y_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "transY")
                trans_y_limit.CreateLowAttr(1.0)
                trans_y_limit.CreateHighAttr(-1.0)

                trans_z_limit = UsdPhysics.LimitAPI.Apply(d6_joint.GetPrim(), "transZ")
                trans_z_limit.CreateLowAttr(1.0)
                trans_z_limit.CreateHighAttr(-1.0)

                # Add joint damping to reduce oscillation at joints
                # This creates more realistic cable behavior
                drive_x = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "rotX")
                drive_x.CreateTargetVelocityAttr(0.0)
                drive_x.CreateDampingAttr(10.0)   # Joint damping coefficient
                drive_x.CreateStiffnessAttr(1.0)  # Spring stiffness
                # drive_x.CreateMaxForceAttr(25000.0)

                drive_y = UsdPhysics.DriveAPI.Apply(d6_joint.GetPrim(), "rotY")
                drive_y.CreateTargetVelocityAttr(0.0)
                drive_y.CreateDampingAttr(10.0)   # Joint damping coefficient
                drive_y.CreateStiffnessAttr(1.0)  # Spring stiffness
                # drive_y.CreateMaxForceAttr(25000.0)
        print(f"Cable created with {num_segments} segments at {base_path}")
        return base_path

    def _remove_cable(self):
        """
        Removes the cable from the stage if it exists.
        """
        if self._cable_path is not None:
            stage = get_current_stage()
            if stage:
                prim = stage.GetPrimAtPath(self._cable_path)
                if prim.IsValid():
                    stage.RemovePrim(self._cable_path)
                    print(f"Cable removed from {self._cable_path}")
            self._cable_path = None

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.

        In this example, a new stage is loaded explicitly, and all assets are reloaded.
        If the user is relying on hot-reloading and does not want to reload assets every time,
        they may perform a check here to see if their desired assets are already on the stage,
        and avoid loading anything if they are.  In this case, the user would still need to add
        their assets to the World (which has low overhead).  See commented code section in this function.
        """
        # Load the UR10e
        # robot_prim_path = "/ur10e"
        # path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"

        # Do not reload assets when hot reloading.  This should only be done while extension is under development.
        # if not is_prim_path_valid(robot_prim_path):
        #     create_new_stage()
        #     add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        # else:
        #     print("Robot already on Stage")

        # create_new_stage()
        # self._add_light_to_stage()
        # add_reference_to_stage(path_to_robot_usd, robot_prim_path)

        # Create a cuboid
        # self._cuboid = FixedCuboid(
        #     "/Scenario/cuboid", position=np.array([0.3, 0.3, 0.5]), size=0.05, color=np.array([255, 0, 0])
        # )

        # self._articulation = SingleArticulation(robot_prim_path)
        
        # Create cable using the parameters
        self._cable_path = self._create_cable(
            base_path="/World/Cable",
            start_position=[0.5, 0, 1.5],
            num_segments=self._float_param_1,
            segment_length=self._float_param_2,
            radius=self._float_param_3
        )

        # Add user-loaded objects to the World
        # world = World.instance()
        # world.scene.add(self._articulation)
        # world.scene.add(self._cuboid)

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.

        In this example, a scenario is initialized which will move each robot joint one at a time in a loop while moving the
        provided prim in a circle around the robot.
        """
        # self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._articulation, self._cuboid)

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        # Remove existing cable
        self._remove_cable()
        
        # self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        Args:
            step (float): The dt of the current physics step
        """
        self._scenario.update_scenario(step)

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False

    #---------------------Additional Callbacks For UI Elements---------------------#

    def _on_float_param_1_changed(self, value: float):
        """Callback when Number of Segments is changed"""
        self._float_param_1 = max(2, value)  # Minimum 2 segments
        print(f"Number of segments changed to: {int(self._float_param_1)}")

    def _on_float_param_2_changed(self, value: float):
        """Callback when Segment Length is changed"""
        self._float_param_2 = max(0.01, value)  # Minimum 1cm
        print(f"Segment length changed to: {self._float_param_2}m")

    def _on_float_param_3_changed(self, value: float):
        """Callback when Capsule Radius is changed"""
        self._float_param_3 = max(0.001, value)  # Minimum 1mm
        print(f"Capsule radius changed to: {self._float_param_3}m")
        
    #---------------------End of Additional Callbacks For UI Elements---------------------#
