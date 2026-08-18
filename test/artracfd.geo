#------------------------------------------------------------------------------
#                                                                             -
#                            Geometry Configuration                           -
#                                                                             -
# - Coordinate system: Right-handed Cartesian system. X-Y plane is the screen -
#   plane; X is horizontal from west to east; Y is vertical from south to     -
#   north; Z axis is perpendicular to screen and points from front to back.   -
# - Coordinates and physical quantities should be normalized to dimensionless.-
# - In each 'begin end' environment, there should NOT be any empty or comment -
#   lines. Please double check input.                                         -
#                                                                             -
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#
#                  >> Number of Geometries <<
#
#------------------------------------------------------------------------------
count begin
1                  # analytical polyhedron (int)
1                  # triangulated polyhedron (int)
count end
#------------------------------------------------------------------------------
#
#                   >> Geometry Information <<
#
# State begin
# O, r, V, W, rho, T, cf, area, volume, mid
# at, ar, ate, g, are, to
# State end
# (Ox, Oy, Oz): geometric center; relative frame for transformation
# r: bounding sphere radius
# (Vx, Vy, Vz): translational velocity of geometric center
# (Wx, Wy, Wz): rotational velocity relative to geometric center
# rho: density; > 1.0e10 if ignore surface force effect
# T: wall temperature; < 0 if adiabatic; >= 0 if constant
# cf: roughness; <= 0 if slip; > 0 if no slip
# area, volume, mid: surface area, volume, material identifier
# (atx, aty, atz): translational acceleration
# (arx, ary, arz): rotational acceleration
# (atex, atey, atez): exerted external translational acceleration
# (gx, gy, gz): gravitational acceleration
# (arex, arey, arez): exerted external rotational acceleration
# to: time to end external ate and are; <= 0 if never end
# stationary object: V = 0; W = 0; ate = 0; g = 0; are = 0; rho > 1.0e36;
#
#------------------------------------------------------------------------------
#
#                 >> Analytical Polyhedron Section <<
#
#------------------------------------------------------------------------------
#sphere state begin
#0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 2700, -1, 1, 0, 0, 0
#0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#sphere state end
#------------------------------------------------------------------------------
#
#                 >> Triangulated Polyhedron Section <<
#
#Polyhedron representation is consistently employed for describing irregular   
#objects. For problems with a collapsed dimension, polyhedron is generated via 
#extending the polygon on the collapsed dimension with unit thickness.         
#------------------------------------------------------------------------------
polyhedron geometry begin
artracfd.stl       # geometry file name
polyhedron geometry end
polyhedron state begin
0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 2700, -1, 1, 0, 0, 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
polyhedron state end
polyhedron transform begin
1, 1, 1, 0, 0, 0, 0, 0, 0 # scale, rotate, translate
polyhedron transform end
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#/* a good practice: end file with a newline */

