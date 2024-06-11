// -----------------------------------------------------------------------------
//
// Backward facing step mesh
//
// -----------------------------------------------------------------------------

// As usual, we start by defining some variables:

Lc1 = 0.125;
x1 = 2.0; xL = 6.0;
yb = -1.0; yt = 1.0;

Point(1) = {0, 0, 0, Lc1};
Point(2) = {x1, 0, 0, Lc1};
Point(3) = {x1, yb, 0, Lc1};
Point(4) = {xL, yb, 0, Lc1};
Point(5) = {xL, yt, 0, Lc1};
Point(6) = {0, yt, 0, Lc1};

Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 5};
Line(5)  = {5, 6};
Line(6)  = {6, 1};

// The third elementary entity is the surface. In order to define a simple
// rectangular surface from the four curves defined above, a curve loop has
// first to be defined. A curve loop is also identified by a tag (unique amongst
// curve loops) and defined by an ordered list of connected curves, a sign being
// associated with each curve (depending on the orientation of the curve to form
// a loop):

Curve Loop(1) = {1, 2, 3, 4, 5, 6};

// We can then define the surface as a list of curve loops (only one here,
// representing the external contour, since there are no holes--see `t4.geo' for
// an example of a surface with a hole):

Plane Surface(1) = {1};

Physical Curve(1) = {6};
Physical Curve(2) = {1,2,3,5};
Physical Curve(3) = {4};
Physical Surface(1) = {1};

Mesh.Algorithm = 1;
Mesh.AnisoMax = 0;
Mesh.ElementOrder = 3;
Mesh.MshFileVersion = 2.2;
// Mesh.LcIntegrationPrecision = 1.0e-15;
Mesh 2;
Save "bfs.msh";
// Exit;
