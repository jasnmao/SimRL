"""Build the experiment scene from ``models/test.xml``.

The original file is a MuJoCo Studio edit of a full-body skeleton with lots of
extra visual clutter (ghost targets, background walls, a table, axis arrows and
a magenta cursor).  Requirement 1 asks for a *completely empty* environment
that contains nothing but the human model and the two spheres, so this module
strips the clutter and appends:

* ``target_ball``  - mocap body, yellow sphere r = 5 cm   (pointing target)
* ``cursor_ball``  - mocap body, purple sphere r = 2 cm   (hand pointing point)
* ``screen_cam``   - the camera that defines the (invisible) screen

The two balls are *not* free bodies: they are placed with ``mj_kinematics`` so
their absolute world pose is under our control (a free body would be affected
by gravity/contacts, a mocap body is kinematically driven instead).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import config as C
from .kinematics import ScreenRig

# bodies / geoms that are visual clutter in the original file
DROP_BODIES = ["cameramover"] + ["target"] + [f"target{i}" for i in range(1, 14)] + ["cursor"]
DROP_GEOMS = ["tableTop", "background_left", "background_back", "Xdir", "Ydir", "Zdir"]


def _rgb_text(rgb255):
    r, g, b = (float(v) / 255.0 for v in rgb255)
    return f"{r:.6f} {g:.6f} {b:.6f} 1"


def _xml_escape(name: str) -> str:
    return name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def lookat_quat(pos, target) -> np.ndarray:
    """Quaternion (w,x,y,z) of a MuJoCo camera at *pos* looking at *target*.

    MuJoCo cameras look along their local -Z with +X to the right and +Y up; the
    model uses +Y as the world up axis.
    """
    pos = np.asarray(pos, float)
    f = np.asarray(target, float) - pos
    f = f / max(np.linalg.norm(f), 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    n = np.linalg.norm(r)
    if n < 1e-6:                      # looking straight up/down: pick another up
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(r)
    r = r / n
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])   # columns = camera axes in world
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def demo_camera_pose():
    """Position/quaternion of the demo camera declared in the scene XML.

    The azimuth is measured from the +X axis (the direction the person faces)
    towards +Z, and the camera is placed that far around the body, slightly above
    the horizontal plane, looking back at ``DEMO_CAM_LOOKAT``.
    """
    look = np.asarray(C.DEMO_CAM_LOOKAT, float)
    az = np.radians(C.DEMO_CAM_AZIMUTH)
    el = np.radians(C.DEMO_CAM_ELEVATION)
    d = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
    pos = look + C.DEMO_CAM_DISTANCE * d
    return pos, lookat_quat(pos, look)


def build_scene(
    src: str = "models/test.xml",
    dst: str = "assets/scene.xml",
    image_size: int = C.IMAGE_SIZE,
    fovy_deg: float = C.CAM_FOVY_DEG,
) -> str:
    tree = ET.parse(src)
    root = tree.getroot()

    # ---------- mesh directory: dst lives in assets/, src in models/ ----------
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "Geometry/")
    if not os.path.isabs(meshdir):
        meshdir = os.path.join(os.path.dirname(os.path.abspath(src)), meshdir)
    compiler.set("meshdir", meshdir.replace("\\", "/"))

    # ---------- offscreen framebuffer big enough for our renders ----------
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.Element("global")
        visual.insert(0, glob)
    glob.set("offwidth", str(max(1280, image_size)))
    glob.set("offheight", str(max(720, image_size)))

    worldbody = root.find("worldbody")

    # ---------- strip clutter ----------
    for body in list(worldbody.findall("body")):
        if body.get("name") in DROP_BODIES:
            worldbody.remove(body)
    for geom in list(worldbody.findall("geom")):
        if geom.get("name") in DROP_GEOMS:
            worldbody.remove(geom)

    # ---------- the two spheres (mocap => purely kinematic) ----------
    def add_ball(name, radius, rgb, x, y, z):
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": name,
                "mocap": "true",
                "pos": f"{x:.6f} {y:.6f} {z:.6f}",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": name,
                "type": "sphere",
                "size": f"{radius:.6f}",
                "rgba": _rgb_text(rgb),
                "contype": "0",
                "conaffinity": "0",
                "mass": "0.001",
            },
        )
        return body

    add_ball("target_ball", C.TARGET_RADIUS, C.TARGET_COLOR, C.SCREEN_X, C.SCREEN_CENTER_Y, C.SCREEN_CENTER_Z)
    add_ball("cursor_ball", C.CURSOR_RADIUS, C.CURSOR_COLOR, C.SCREEN_X, C.SCREEN_CENTER_Y, C.SCREEN_CENTER_Z)

    # ---------- the screen camera ----------
    # it sits CAM_DISTANCE in front of the screen along +X and looks back at the
    # body; the screen frame's axes give its orientation directly
    R0 = ScreenRig.BASE_ROT
    cam_pos = np.array([C.SCREEN_X, C.SCREEN_CENTER_Y, C.SCREEN_CENTER_Z]) + R0 @ np.array(
        [0.0, 0.0, C.CAM_DISTANCE]
    )
    cam_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cam_quat, R0.flatten())
    cam = ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "screen_cam",
            "pos": " ".join(f"{v:.6f}" for v in cam_pos),
            "quat": " ".join(f"{v:.6f}" for v in cam_quat),
            "fovy": f"{fovy_deg:.6f}",
        },
    )
    # remove stale camera if the source already had one with this name
    cams = [c for c in worldbody.findall("camera") if c is not cam]
    for c in cams:
        if c.get("name") == "screen_cam":
            worldbody.remove(c)

    # ---------- third-person camera for the demo video ----------
    dpos, dquat = demo_camera_pose()
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "demo_cam",
            "pos": " ".join(f"{v:.6f}" for v in dpos),
            "quat": " ".join(f"{v:.6f}" for v in dquat),
            "fovy": "45",
        },
    )

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    return dst


def load_model(path: str = "assets/scene.xml") -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(path)


def ensure_scene(dst: str = "assets/scene.xml", src: str = "models/test.xml") -> str:
    """Build the scene if it is missing or if the screen geometry changed."""
    need = True
    if os.path.exists(dst):
        try:
            m = mujoco.MjModel.from_xml_path(dst)
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "screen_cam")
            need = (
                cid < 0
                or mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam") < 0
                or abs(float(m.cam_fovy[cid]) - C.CAM_FOVY_DEG) > 1e-3
                or abs(float(m.cam_pos[cid][0]) - (C.SCREEN_X + C.CAM_DISTANCE)) > 1e-6
                or abs(float(m.cam_pos[cid][1]) - C.SCREEN_CENTER_Y) > 1e-6
                or abs(float(m.cam_pos[cid][2]) - C.SCREEN_CENTER_Z) > 1e-6
                or int(m.nmocap) != 2
                or int(m.vis.global_.offwidth) != int(max(1280, C.IMAGE_SIZE))
            )
        except Exception:
            need = True
    if need:
        build_scene(src, dst)
    return dst


if __name__ == "__main__":
    p = build_scene()
    m = load_model(p)
    print("wrote", p, "nbody", m.nbody, "njnt", m.njnt, "nu", m.nu, "nmocap", m.nmocap)

