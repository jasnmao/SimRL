"""Central configuration for the pointing experiment.

All physical constants that depend on the human model live here so that the
geometry can be re-derived (see ``calibrate.py``) without touching the code.

Coordinate system of models/test.xml:  Y is up,  X is forward (out of the
chest),  Z is the model's left.  The screen is the plane  x = SCREEN_X  with
the body entirely behind it (x < SCREEN_X).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any

# --------------------------------------------------------------------------
# joints / actuation
# --------------------------------------------------------------------------
JOINTS = [
    "elv_angle",
    "shoulder_elv",
    "shoulder_rot",
    "elbow_flexion",
    "pro_sup",
    "deviation",
    "flexion",
]

# --------------------------------------------------------------------------
# screen rig geometry (calibrated, see calibrate.py / README)
# --------------------------------------------------------------------------
# front-most surface of the chest (thorax/clavicle meshes) is x = +0.09014;
# the chest normal is +X (verified by rendering from +X: sternum visible)
THORAX_FRONT_X = 0.09014
SCREEN_DISTANCE = 0.55                      # requirement: 55 cm in front of the body
SCREEN_X = THORAX_FRONT_X + SCREEN_DISTANCE  # 0.64014 m, plane normal = +X

# size of the (invisible) screen, in metres.  Chosen in calib_screen.py so that the
# reachable region of the pointing hand fills the rendered image nicely.
SCREEN_SIZE = 0.72
# centre of the screen in the plane:  world Y (up/down) and world Z (left/right)
SCREEN_CENTER_Y = -0.30
SCREEN_CENTER_Z = 0.20

# camera sits SCREEN_DISTANCE in front of the screen, looking back at the body
CAM_DISTANCE = SCREEN_DISTANCE
CAM_FOVY_DEG = 2.0 * math.degrees(math.atan(0.5 * SCREEN_SIZE / CAM_DISTANCE))

IMAGE_SIZE = 640                            # rendered + cropped screen image

TARGET_COLOR = (250.0, 215.0, 30.0)         # RGB 0-255
CURSOR_COLOR = (126.0, 34.0, 206.0)
TARGET_RADIUS = 0.05
CURSOR_RADIUS = 0.02

# --------------------------------------------------------------------------
# initial arm pose [rad] in C.JOINTS order
# --------------------------------------------------------------------------
# The stock "Init" keyframe has the arm hanging straight down and outwards, so
# with an +X-facing screen the fingertip would project far outside the image.
# This pose was solved by IK (solve_init_pose.py) so that the elbow is flexed and
# the fingertip points at the middle of the screen, ~12 cm behind its plane.
INIT_Q7 = (-0.31589583, 0.07816682, 0.03589437, 1.34193942, 0.03626085, 0.12666645, 0.12293656,)

# third-person camera used by the demo video: it must show the *whole* human model
# (plus the two balls) so the video is understandable.  It sits in front of the
# person (the screen side, +X) and off to one side, looking back at the body.
DEMO_CAM_LOOKAT = (0.28, -0.30, 0.14)
DEMO_CAM_AZIMUTH = 50.0      # degrees, measured from +X towards +Z
DEMO_CAM_ELEVATION = 11.0    # degrees above the horizontal plane
DEMO_CAM_DISTANCE = 0.95

# --------------------------------------------------------------------------
# swing / sway
# --------------------------------------------------------------------------
@dataclass
class SwayConfig:
    """Moderate sway of the whole screen rig (camera + screen plane together).

    The body itself is *not* perturbed: sway is a rigid motion of the rig about
    the screen centre, which is exactly the visual equivalent of the world
    shaking while the observer keeps pointing at a screen bolted to the world.
    """
    enabled: bool = True
    # rigid translation of the rig, metres (x, y, z world axes)
    trans_amp: Tuple[float, float, float] = (0.020, 0.020, 0.015)
    # rigid rotation of the rig about the screen centre, degrees (x, y, z)
    rot_amp_deg: Tuple[float, float, float] = (2.0, 2.0, 1.5)
    # angular frequency band, Hz  (kept slow: "moderate amplitude")
    freq_range: Tuple[float, float] = (0.25, 0.60)
    # additional 2-D sway of the target on the screen plane, metres
    target_sway_amp: Tuple[float, float] = (0.014, 0.014)
    target_sway_freq: Tuple[float, float] = (0.20, 0.50)


# --------------------------------------------------------------------------
# control / episode
# --------------------------------------------------------------------------
@dataclass
class ControlConfig:
    max_action: float = 0.05      # rad, bound of one joint-position increment
    substeps: int = 20            # 20 * 0.002 s = 40 ms per decision
    action_smooth: str = "step"
    # Joint-space impedance of the servo.  By default the gains are derived from
    # the inertia diagonal at the Init pose: kp = I*omega_n^2, kd = 2*zeta*I*omega_n.
    omega_n: float = 25.0         # rad/s natural frequency of the position loop
    zeta: float = 1.0             # damping ratio (critically damped)
    kp: float | None = None       # explicit override
    kd: float | None = None
    id_clip: bool = True          # clip torques to the actuator limits


@dataclass
class EpisodeConfig:
    max_steps: int = 120
    success_px: float = 20.0          # |x-p| threshold in pixels (~2.2 cm on screen)
    success_qvel: float = 0.50        # every |dq| below this  [rad/s] (~29 deg/s)
    success_hold: int = 3             # consecutive steps satisfying the above
    target_region: Tuple[float, float, float, float] = (-0.13, 0.13, -0.11, 0.11)
    # (xmin, xmax, ymin, ymax) on the screen plane, metres relative to its centre
    # the target is never drawn closer than this to the starting cursor [px],
    # otherwise an episode could start already inside the success region
    target_min_start_px: float = 120.0


# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------
@dataclass
class RewardConfig:
    r1: float = 0.005     # torque penalty (auxiliary, small)
    r2: float = 0.002     # acceleration penalty (auxiliary, smaller)
    r3: float = 0.05      # velocity penalty: teaches the policy to *stop* on target
    hold_bonus: float = 1.0        # big per-step bonus while pointing *and* stopped
    success_bonus: float = 0.0     # extra terminal bonus (now redundant)
    dist_scale_px: float = 200.0   # normalisation of |x-p|
    tau_scale: float = 20.0        # N m, normalisation of ||tau||
    qdd_scale: float = 40.0        # rad/s^2, normalisation of ||ddq||
    qvel_scale: float = 1.0        # rad/s, normalisation of ||dq||


# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------
@dataclass
class PPOConfig:
    num_envs: int = 8
    rollout_steps: int = 64           # per env
    total_updates: int = 220
    gamma: float = 0.985
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 256
    lr: float = 3e-4
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: Tuple[int, ...] = (128, 128)
    log_std_init: float = -1.0
    device: str = "cuda"


@dataclass
class Config:
    xml: str = "assets/scene.xml"
    raw_xml: str = "models/test.xml"
    det_weights: str = "yolo_model/best.pt"
    det_conf: float = 0.25
    det_conf_min: float = 0.20    # below this the analytic centre is used instead
    # plausible apparent diameter (px) of each sphere; a box outside this range means
    # the detector's regression branch misfired and the analytic centre is used
    det_size_range: tuple = ((45.0, 160.0), (15.0, 80.0))
    det_use_cuda_graph: bool = False
    det_device: str = "cuda"
    det_fallback: bool = True     # analytic centre when detection is missing
    obs_vel_scale: float = 5.0    # rad/s, observation normalisation
    obs_acc_scale: float = 150.0  # rad/s^2
    out_dir: str = "log"
    seed: int = 0
    sway: SwayConfig = field(default_factory=SwayConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_config() -> Config:
    return Config()
