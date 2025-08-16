# Usage:
# blender -b -P rigify_autorig.py -- --in /path/to/mesh.glb --out /path/to/rigged.fbx
import bpy, sys, os, argparse, math

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_mesh", required=True)
    parser.add_argument("--out", dest="out_fbx", required=True)
    return parser.parse_args(argv)

def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext in [".obj"]:
        bpy.ops.import_scene.obj(filepath=path)
    else:
        raise RuntimeError(f"Unsupported mesh format: {ext}")

def get_mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]

def add_metarig_and_fit(mesh_obj):
    # Enable Rigify
    bpy.ops.preferences.addon_enable(module="rigify")
    # Add human meta-rig
    bpy.ops.object.armature_human_metarig_add()
    rig = bpy.context.active_object
    # Rough fit based on mesh bounding box
    minx, miny, minz = [min(v[i] for v in mesh_obj.bound_box) for i in range(3)]
    maxx, maxy, maxz = [max(v[i] for v in mesh_obj.bound_box) for i in range(3)]
    height = maxz - minz
    mesh_center = ((minx+maxx)/2, (miny+maxy)/2, minz)
    rig.location = mesh_center
    # Scale metarig to mesh height (heuristic)
    if height > 0:
        rig.scale = (height/1.8, height/1.8, height/1.8)
    bpy.context.view_layer.update()
    return rig

def parent_with_auto_weights(mesh_obj, armature_obj):
    # Select mesh, then armature
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    mesh_obj.select_set(True)
    # Parent with automatic weights
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

def generate_rig(armature_obj):
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    try:
        bpy.ops.pose.rigify_generate()
    except Exception:
        # If generation fails, keep metarig as the armature
        pass

def export_fbx(path):
    bpy.ops.export_scene.fbx(filepath=path, use_selection=False, apply_scale_options='FBX_SCALE_UNITS', bake_space_transform=True)

def main():
    args = parse_args()
    clean_scene()
    import_mesh(args.in_mesh)
    meshes = get_mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh imported")
    mesh = meshes[0]
    rig = add_metarig_and_fit(mesh)
    parent_with_auto_weights(mesh, rig)
    # Try generating a control rig; if not, leave as simple armature
    generate_rig(rig)
    export_fbx(args.out_fbx)

if __name__ == "__main__":
    main()