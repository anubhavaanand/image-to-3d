import trimesh
from ..engines.base import Mesh3D

def export_glb(mesh: Mesh3D, filepath: str) -> None:
    """Exports a Mesh3D object to a GLB file."""
    if mesh is None:
        raise ValueError("Mesh3D is None, cannot export.")
        
    t_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=mesh.vertex_colors
    )
    t_mesh.export(filepath, file_type='glb')
