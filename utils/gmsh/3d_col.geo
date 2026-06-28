// Gmsh project created on Tue Mar 26 09:53:47 2024
SetFactory("OpenCASCADE");

l = 1.0;
w = 1.0;
h = 4.0;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {l, 0, 0, 1.0};
Point(3) = {l, w, 0, 1.0};
Point(4) = {0, w, 0, 1.0};
Point(5) = {0, 0, h, 1.0};
Point(6) = {l, 0, h, 1.0};
Point(7) = {l, w, h, 1.0};
Point(8) = {0, w, h, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {5, 1};
Line(10) = {6, 2};
Line(11) = {7, 3};
Line(12) = {8, 4};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Curve Loop(3) = {1, 10, 5, 9};
Curve Loop(4) = {2, 11, 6, 10};
Curve Loop(5) = {3, 12, 7, 11};
Curve Loop(6) = {4, 9, 8, 12};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};

Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};
Physical Surface(5) = {5};
Physical Surface(6) = {6};

Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};
Physical Volume(1) = {1};

Mesh.Algorithm3D = 1;
Mesh.AnisoMax = 0;
Mesh.MshFileVersion = 2.2;
//Mesh.MeshSizeFactor = 0.25;
Mesh 3; // This is important
Save "3d_col.msh";
Exit;
