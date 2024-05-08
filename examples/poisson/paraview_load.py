### Set the root directory where paraview_output_0, ... are located.
root_dir = '/absolute/path/to/your/paraview/output'

# trace generated using paraview version 5.11.2
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

import os

prefix = "paraview_output"
os.chdir(root_dir)
filenames = [f for f in os.listdir('.') if os.path.isdir(f) and '_'.join(f.split('_')[0:-1]) == prefix]
Nk = len(filenames)

paraview_output_pvd = []
paraview_output_pvdDisplay = []

for k in range(Nk):
    # create a new 'PVD Reader'
    paraview_output_pvd += [PVDReader(registrationName='%s.pvd' % filenames[k], FileName='%s/%s/%s.pvd' % (root_dir, filenames[k], filenames[k]))]
    paraview_output_pvd[-1].CellArrays = ['attribute']
    paraview_output_pvd[-1].PointArrays = ['solution']

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    paraview_output_pvdDisplay += [Show(paraview_output_pvd[-1], renderView1, 'UnstructuredGridRepresentation')]

    # trace defaults for the display properties.
    paraview_output_pvdDisplay[-1].Representation = 'Surface'
    paraview_output_pvdDisplay[-1].ColorArrayName = [None, '']
    paraview_output_pvdDisplay[-1].SelectTCoordArray = 'None'
    paraview_output_pvdDisplay[-1].SelectNormalArray = 'None'
    paraview_output_pvdDisplay[-1].SelectTangentArray = 'None'
    paraview_output_pvdDisplay[-1].OSPRayScaleArray = 'solution'
    paraview_output_pvdDisplay[-1].OSPRayScaleFunction = 'PiecewiseFunction'
    paraview_output_pvdDisplay[-1].SelectOrientationVectors = 'None'
    paraview_output_pvdDisplay[-1].ScaleFactor = 0.1
    paraview_output_pvdDisplay[-1].SelectScaleArray = 'None'
    paraview_output_pvdDisplay[-1].GlyphType = 'Arrow'
    paraview_output_pvdDisplay[-1].GlyphTableIndexArray = 'None'
    paraview_output_pvdDisplay[-1].GaussianRadius = 0.005
    paraview_output_pvdDisplay[-1].SetScaleArray = ['POINTS', 'solution']
    paraview_output_pvdDisplay[-1].ScaleTransferFunction = 'PiecewiseFunction'
    paraview_output_pvdDisplay[-1].OpacityArray = ['POINTS', 'solution']
    paraview_output_pvdDisplay[-1].OpacityTransferFunction = 'PiecewiseFunction'
    paraview_output_pvdDisplay[-1].DataAxesGrid = 'GridAxesRepresentation'
    paraview_output_pvdDisplay[-1].PolarAxes = 'PolarAxesRepresentation'
    paraview_output_pvdDisplay[-1].ScalarOpacityUnitDistance = 0.10171193339768637
    paraview_output_pvdDisplay[-1].OpacityArrayName = ['POINTS', 'solution']
    paraview_output_pvdDisplay[-1].SelectInputVectors = [None, '']
    paraview_output_pvdDisplay[-1].WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    paraview_output_pvdDisplay[-1].ScaleTransferFunction.Points = [-0.0030751066823493183, 0.0, 0.5, 0.0, 0.002953424556242068, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    paraview_output_pvdDisplay[-1].OpacityTransferFunction.Points = [-0.0030751066823493183, 0.0, 0.5, 0.0, 0.002953424556242068, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera(False)

    #changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.5, 0.5, 3.35]
    renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(paraview_output_pvdDisplay[-1], ('POINTS', 'solution'))

    # rescale color and/or opacity maps used to include current data range
    paraview_output_pvdDisplay[-1].RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    paraview_output_pvdDisplay[-1].SetScalarBarVisibility(renderView1, True)

# get 2D transfer function for 'solution'
solutionTF2D = GetTransferFunction2D('solution')

# get color transfer function/color map for 'solution'
solutionLUT = GetColorTransferFunction('solution')
solutionLUT.TransferFunction2D = solutionTF2D
solutionLUT.RGBPoints = [-0.0030751066823493183, 0.231373, 0.298039, 0.752941, -6.08410630536251e-05, 0.865003, 0.865003, 0.865003, 0.002953424556242068, 0.705882, 0.0156863, 0.14902]
solutionLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'solution'
solutionPWF = GetOpacityTransferFunction('solution')
solutionPWF.Points = [-0.0030751066823493183, 0.0, 0.5, 0.0, 0.002953424556242068, 1.0, 0.5, 0.0]
solutionPWF.ScalarRangeInitialized = 1

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(2410, 1482)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5, 0.5, 3.35]
renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
renderView1.CameraParallelScale = 0.7071067811865476

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).