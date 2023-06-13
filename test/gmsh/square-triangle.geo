// -----------------------------------------------------------------------------
//
//  Gmsh GEO tutorial 4
//
//  Built-in functions, holes in surfaces, annotations, entity colors
//
// -----------------------------------------------------------------------------

// As usual, we start by defining some variables:

Lc1 = 0.125; Lc2 = 0.04;

Point(1) = {0, 0, 0, Lc1}; Point(2) = {1, 0, 0, Lc1};
Point(3) = {0, 1, 0, Lc1}; Point(4) = {1, 1, 0, Lc1};

Line(1)  = {1 , 2};
Line(2)  = {2 , 4};
Line(3)  = {4 , 3};
Line(4)  = {3 , 1};

nc = 3; cangle = (nc - 2) * Pi / nc;
cx = 0.5; cy = 0.5; r0 = 0.25;
r1 = 0.05; r2 = r0 - r1 / Cos(0.5 * cangle); r3 = r1 * Tan(0.5 * cangle);
For i In {0:nc-1}
    angle1 = i * 2. * Pi / nc + 1.0 * Pi;
    ix = cx + r2 * Cos(angle1);
    iy = cy + r2 * Sin(angle1);
    Point(i+5) = {ix, iy, 0, Lc2};

    angle2 = 0.5 * Pi - 0.5 * cangle;
    Point((2*i+1)+5+(nc-1)) = {ix + r3 * Cos(angle1 - angle2), iy + r3 * Sin(angle1 - angle2), 0, Lc2};
    Point((2*i+2)+5+(nc-1)) = {ix + r3 * Cos(angle1 + angle2), iy + r3 * Sin(angle1 + angle2), 0, Lc2};
EndFor

maxIdx = 2*(nc-1)+2+5+(nc-1);
For i In {0:nc-1}
    Circle(2*i+5) = {(2*i+1)+5+(nc-1),i+5,(2*i+2)+5+(nc-1)};
    idx1 = (2*i+2)+5+(nc-1);
    idx2 = idx1+1;
    If (idx2 > maxIdx)
        idx2 -= 2*nc;
    EndIf
    Line(2*i+6) = {idx1, idx2};
EndFor

// The third elementary entity is the surface. In order to define a simple
// rectangular surface from the four curves defined above, a curve loop has
// first to be defined. A curve loop is also identified by a tag (unique amongst
// curve loops) and defined by an ordered list of connected curves, a sign being
// associated with each curve (depending on the orientation of the curve to form
// a loop):

Curve Loop(1) = {4, 1, 2, 3};
Curve Loop(2) = {5:5+(2*nc-1)};

// We can then define the surface as a list of curve loops (only one here,
// representing the external contour, since there are no holes--see `t4.geo' for
// an example of a surface with a hole):

Plane Surface(1) = {1, 2};

Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
Physical Curve(5) = {5:5+(2*nc-1)};
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
Save "square-triangle.msh";
// Exit;
