import stltovoxel
stl_path = './dev_scripts/voxelizer/squirtle.stl'
output_path = "./dev_scripts/voxelizer/sliced_squirtle/s.png"
stltovoxel.convert_file(stl_path, output_path, resolution=47)