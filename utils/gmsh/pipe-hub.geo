// -----------------------------------------------------------------------------
//
//  Gmsh GEO tutorial 4
//
//  Built-in functions, holes in surfaces, annotations, entity colors
//
// -----------------------------------------------------------------------------

// As usual, we start by defining some variables:

Lc1 = 0.125;

Point(1) = {0, 0, 0, Lc1}; Point(2) = {1, 0, 0, Lc1};
Point(3) = {0, 1, 0, Lc1}; Point(4) = {1, 1, 0, Lc1};

Point(5) = {0.25, 0, 0, Lc1}; Point(6) = {0.75, 0, 0, Lc1};
Point(7) = {1, 0.25, 0, Lc1}; Point(8) = {1, 0.75, 0, Lc1};
Point(9) = {0.75, 1, 0, Lc1}; Point(10) = {0.25, 1, 0, Lc1};
Point(11) = {0, 0.75, 0, Lc1}; Point(12) = {0, 0.25, 0, Lc1};

Line(1)  = {5 , 6};
Circle(2) = {6,2,7};
Line(3)  = {7 , 8};
Circle(4) = {8,4,9};
Line(5)  = {9 , 10};
Circle(6) = {10,3,11};
Line(7)  = {11 , 12};
Circle(8) = {12,1,5};

// The third elementary entity is the surface. In order to define a simple
// rectangular surface from the four curves defined above, a curve loop has
// first to be defined. A curve loop is also identified by a tag (unique amongst
// curve loops) and defined by an ordered list of connected curves, a sign being
// associated with each curve (depending on the orientation of the curve to form
// a loop):

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};

// We can then define the surface as a list of curve loops (only one here,
// representing the external contour, since there are no holes--see `t4.geo' for
// an example of a surface with a hole):

Plane Surface(1) = {1};

Physical Curve(1) = {1};
Physical Curve(2) = {3};
Physical Curve(3) = {5};
Physical Curve(4) = {7};
Physical Curve(5) = {2, 4, 6, 8};
Physical Surface(1) = {1};

// Point(10) = {0.2, 0, 0, Lc1};
// Point(11) = {0.4, 0, 0, Lc1};
// Point(12) = {0.6, 0, 0, Lc1};
// Point(13) = {0.8, 0, 0, Lc1};

// Point{10} In Surface {1};
// Point{11} In Surface {1};
// Point{12} In Surface {1};
// Point{13} In Surface {1};

Mesh.Algorithm = 1;
Mesh.AnisoMax = 0;
Mesh.ElementOrder = 3;
Mesh.MshFileVersion = 2.2;
// Mesh.LcIntegrationPrecision = 1.0e-15;
Mesh 2;
Save "pipe-hub.msh";
// Exit;
