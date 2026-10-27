"""Optional host-only spectator. Never used for control or evaluation.

Imported only with --spectator; standalone benchmark has no Viser dependency.
Physics and USD reads stay on the simulation thread.
"""
import logging
import socket
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf


class SceneViewer:
    def __init__(self, stage):
        import viser
        import trimesh
        self.stage = stage
        self.nodes = []
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        self.server.initial_camera.position = (2.1, 1.8, 2.1)
        self.server.initial_camera.look_at = (0.65, -0.05, 1.1)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.gui.configure_theme(control_layout="floating", show_logo=False)
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            kind = prim.GetTypeName()
            if kind not in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"):
                continue
            imageable = UsdGeom.Imageable(prim)
            # Keep currently invisible prims too: instrument displays (seven-segment
            # timer digits, state labels) change by toggling USD visibility.
            if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            try:
                if kind == "Mesh":
                    mesh = UsdGeom.Mesh(prim)
                    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                    counts = mesh.GetFaceVertexCountsAttr().Get()
                    indices = mesh.GetFaceVertexIndicesAttr().Get()
                    faces, offset = [], 0
                    for count in counts:
                        faces.extend((indices[offset], indices[offset + j], indices[offset + j + 1])
                                     for j in range(1, count - 1))
                        offset += count
                    faces = np.asarray(faces, dtype=np.uint32)
                else:
                    attr = lambda name, default: prim.GetAttribute(name).Get() or default
                    if kind == "Cube":
                        mesh = trimesh.creation.box(extents=[float(attr("size", 2))] * 3)
                    elif kind == "Sphere":
                        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(attr("radius", 1)))
                    else:
                        radius, height = float(attr("radius", 1)), float(attr("height", 2))
                        if kind == "Capsule":
                            mesh = trimesh.creation.capsule(radius=radius, height=height)
                        elif kind == "Cone":
                            mesh = trimesh.creation.cone(radius=radius, height=height, sections=32)
                            mesh.vertices[:, 2] -= height / 2
                        else:
                            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
                        axis = attr("axis", "Z")
                        if axis == "X":
                            mesh.vertices = mesh.vertices[:, [2, 1, 0]]
                        elif axis == "Y":
                            mesh.vertices = mesh.vertices[:, [0, 2, 1]]
                    vertices, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32)
                if not len(vertices) or not len(faces):
                    continue
                color = self.color(prim)
                handle = self.server.scene.add_mesh_simple(
                    f"/scene/mesh_{len(self.nodes)}", vertices, faces, color=color, side="double")
                self.nodes.append((prim, handle, [None]))
            except Exception:
                logging.exception("Viser could not mirror %s", prim.GetPath())
        if not self.nodes:
            self.server.stop()
            raise RuntimeError("Viser scene contains no visible geometry")
        self.update()

    @staticmethod
    def color(prim):
        display = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        if display:
            return tuple(int(np.clip(v, 0, 1) * 255) for v in display[0])
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            for node in Usd.PrimRange(material.GetPrim()):
                for name in ("inputs:diffuse_color_constant", "inputs:diffuseColor", "inputs:base_color"):
                    value = node.GetAttribute(name).Get()
                    if value is not None:
                        return tuple(int(np.clip(v, 0, 1) * 255) for v in value[:3])
        return (170, 180, 190)

    def update(self):
        cache = UsdGeom.XformCache()
        with self.server.atomic():
            for prim, handle, shown in self.nodes:
                visible = UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
                if visible != shown[0]:
                    handle.visible = visible
                    shown[0] = visible
                if not visible:
                    continue
                transform = Gf.Transform(cache.GetLocalToWorldTransform(prim))
                quat = transform.GetRotation().GetQuat()
                handle.position = tuple(transform.GetTranslation())
                handle.wxyz = (quat.GetReal(), *quat.GetImaginary())
                handle.scale = tuple(transform.GetScale())
        self.server.flush()

    def metadata(self):
        return {"port": self.server.get_port(), "read_only": True,
                "geometry_count": len(self.nodes), "source": "live-usd"}

    def close(self):
        self.server.stop()
