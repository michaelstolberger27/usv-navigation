"""
VesselFactory that creates YP-model vessels with the HybridAutomatonController.
"""

import copy
import numpy as np

from commonocean.scenario import obstacle
from commonocean.scenario.obstacle import ObstacleType
from commonocean.common.solution import VesselModel, VesselType
from commonocean.scenario.state import State
from commonocean_dc.feasibility.vessel_dynamics import VesselDynamics
from Environment.SurfaceVessel import SurfaceVessel
from Environment.VesselFactory import VesselFactory
from Util.Utilities import divide_long_waypointpaths

from .controller import HybridAutomatonController


class ColavVesselFactory(VesselFactory):
    """
    Factory that creates YP-model vessels controlled by the
    HybridAutomatonController (prescribed-time COLAV automaton).
    """

    def __init__(
        self,
        dt: float,
        current_configuration: dict,
        *,
        a: float = 1.67,
        v: float = 12.0,
        eta: float = 3.5,
        tp: float = 1.0,
        Cs: float = 2.0,
        v1_buffer: float = 0.0,
        vessel_type: VesselType = VesselType.Vessel1,
    ):
        super().__init__(dt, current_configuration)
        self.current_configuration = copy.deepcopy(current_configuration)
        self.vessel_type = vessel_type

        # Controller parameters — passed to every vessel
        self.ctrl_params = dict(
            a=a, v=v, eta=eta, tp=tp, Cs=Cs, v1_buffer=v1_buffer,
        )

    def get_parameters(self):
        return self.ctrl_params

    def get_vessel(
        self,
        init_state: State,
        waypoints: np.ndarray,
        dt: float,
        ship_name: str = None,
        paramid=None,
        vesselid=None,
        goal_waypoint: tuple = None,
    ) -> SurfaceVessel:
        # Vessel dynamics (YP model)
        dynamics = VesselDynamics.from_model(VesselModel.YP, self.vessel_type)

        # Dynamic obstacle (collision geometry)
        dyn_id = self.generate_id()
        dynamic_obstacle = obstacle.DynamicObstacle(
            dyn_id, ObstacleType.UNKNOWN,
            dynamics.shape, init_state, None,
        )

        # Surface vessel
        vid = self.generate_id() if vesselid is None else vesselid
        vessel = SurfaceVessel(dynamic_obstacle, dynamics, vid, dt)
        vessel.current_configuration = self.current_configuration
        vessel.id = vid

        # Waypoints (subdivide long legs)
        max_ts = self.current_configuration[
            "scenario_pre_processing"
        ]["expected_maximum_ts_per_wp"]
        waypoints = divide_long_waypointpaths(
            waypoints, dynamics.parameters.v_max * dt * max_ts,
        )
        vessel.set_waypoints(waypoints)

        # Controller
        ctrl_kwargs = {**self.ctrl_params}
        if goal_waypoint is not None:
            ctrl_kwargs['goal_waypoint'] = goal_waypoint
        controller = HybridAutomatonController(vessel, dt, **ctrl_kwargs)
        controller.initialise()
        vessel.set_controller(controller)

        # Name
        vessel.vessel_name = ship_name or f"COLAV_{VesselFactory.vessel_count}"
        VesselFactory.vessel_count += 1
        return vessel
