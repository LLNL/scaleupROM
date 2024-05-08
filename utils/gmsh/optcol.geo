SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.2, 0, 0, 1.0};
//+
Point(3) = {0.3, 2, 0, 1.0};
//+
Point(4) = {0.2, 4, 0, 1.0};
//+
Point(5) = {0, 4, 0, 1.0};
//+
Point(6) = {-0.1, 2, 0, 1.0};

//+
Line(1) = {1, 2};
//+
Spline(2) = {2, 3, 4};
//+
Line(3) = {4, 5};
//+
Spline(4) = {5, 6, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};

Physical Curve(1) = {1};
Physical Curve(2) = {2};
Physical Curve(3) = {3};
Physical Curve(4) = {4};
Physical Surface(1) = {1};

Mesh.Algorithm = 1;
Mesh.AnisoMax = 0;
Mesh.MshFileVersion = 2.2;
Mesh 2; // This is important
RefineMesh;
RefineMesh;
Save "optcol.msh";
Exit;