// Gmsh project created on Tue Mar 26 09:53:47 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.2, 0, 0, 1.0};
//+
Point(3) = {0.2, 0.2, 0, 1.0};
//+
Point(4) = {0, 0.2, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
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
Save "optjoint.msh";
Exit;