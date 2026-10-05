# wind-turbine-rotor-cfd

Transient OpenFOAM simulation of a three-bladed model wind-turbine rotor. The rotor
turns through the mesh on a sliding interface (AMI), resolving the tip vortices, the
wake and the slow-down of the air ahead of the blades.

<!-- Drag rotor.mp4 into this README in the GitHub editor to embed the animation here. -->

## Requirements

* OpenFOAM v2306 or newer (openfoam.com)
* Python 3 with numpy and matplotlib for the load plots; pyvista, pillow and ffmpeg for
  the animation

## Usage

```sh
cd case
./Allrun -preset coarse -np 8       # mesh, solve, plot loads
python3 ../post/render_movie.py .   # animation of the last revolution
./Allclean
```

Wind speed and tip speed ratio are set in `case/system/caseParameters`. Mesh resolution
is chosen with `-preset coarse`, `medium` or `fine`.

## Credits

Blade chord and twist from Krogstad & Eriksen (2013), *Renewable Energy* 50, 325-333,
as tabulated by P. Bachant (CC-BY 4.0). S826 airfoil coordinates from Somers (2005),
NREL/SR-500-36344.

## License

MIT, see `LICENSE`.
