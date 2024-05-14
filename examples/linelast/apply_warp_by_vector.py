# trace generated using paraview version 5.11.2
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11


def apply_warpvector(filename):
    # find source
    srcsol = FindSource(filename)
    # set active source
    SetActiveSource(srcsol)

    # get color transfer function/color map for 'solution'
    solutionLUT = GetColorTransferFunction('solution')

    # get opacity transfer function/opacity map for 'solution'
    solutionPWF = GetOpacityTransferFunction('solution')

    # get 2D transfer function for 'solution'
    solutionTF2D = GetTransferFunction2D('solution')

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # get display properties
    srcsolDisplay = GetDisplayProperties(srcsol, view=renderView1)

    # create a new 'Warp By Vector'
    warpByVector = WarpByVector(registrationName='WarpByVector_'+filename, Input=srcsol)
    warpByVector.Vectors = ['POINTS', 'solution']

    # set active source
    SetActiveSource(warpByVector)

    # show data in view
    warpByVectorDisplay = Show(warpByVector, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    warpByVectorDisplay.Representation = 'Surface'
    warpByVectorDisplay.ColorArrayName = ['POINTS', 'solution']
    warpByVectorDisplay.LookupTable = solutionLUT
    warpByVectorDisplay.SelectTCoordArray = 'None'
    warpByVectorDisplay.SelectNormalArray = 'None'
    warpByVectorDisplay.SelectTangentArray = 'None'
    warpByVectorDisplay.OSPRayScaleArray = 'solution'
    warpByVectorDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    warpByVectorDisplay.SelectOrientationVectors = 'None'
    warpByVectorDisplay.ScaleFactor = 1.0034994557994217
    warpByVectorDisplay.SelectScaleArray = 'None'
    warpByVectorDisplay.GlyphType = 'Arrow'
    warpByVectorDisplay.GlyphTableIndexArray = 'None'
    warpByVectorDisplay.GaussianRadius = 0.05017497278997109
    warpByVectorDisplay.SetScaleArray = ['POINTS', 'solution']
    warpByVectorDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    warpByVectorDisplay.OpacityArray = ['POINTS', 'solution']
    warpByVectorDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    warpByVectorDisplay.DataAxesGrid = 'GridAxesRepresentation'
    warpByVectorDisplay.PolarAxes = 'PolarAxesRepresentation'
    warpByVectorDisplay.ScalarOpacityFunction = solutionPWF
    warpByVectorDisplay.ScalarOpacityUnitDistance = 1.2815333878254977
    warpByVectorDisplay.OpacityArrayName = ['POINTS', 'solution']
    warpByVectorDisplay.SelectInputVectors = ['POINTS', 'solution']
    warpByVectorDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    warpByVectorDisplay.ScaleTransferFunction.Points = [-0.01822456776713409, 0.0, 0.5, 0.0, 0.01827591453119054, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    warpByVectorDisplay.OpacityTransferFunction.Points = [-0.01822456776713409, 0.0, 0.5, 0.0, 0.01827591453119054, 1.0, 0.5, 0.0]

    # show color bar/color legend
    warpByVectorDisplay.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(srcsol, renderView1)

#### import the simple module from the paraview
from paraview.simple import *
import os
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

root_dir = '/Users/larsson4/repos/scaleupROM/build/examples/linelast'
prefix = "paraview_output"
os.chdir(root_dir)
filenames = [str(f)+'.pvd' for f in os.listdir('.') if os.path.isdir(f) and '_'.join(f.split('_')[0:-1]) == prefix]
for filename in filenames:
    print(filename)
    apply_warpvector(filename)
