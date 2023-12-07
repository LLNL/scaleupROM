# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'PVD Reader'
ncomp = 128
offset = 128
rootDir = 'C:\\Users\\chung28\\Desktop\\slides\\stokes_array\\stokes_array_32x32_rom'
prefix = 'stokes_channel_output'
filenames = ['%s_%d' % (prefix, n + offset) for n in range(ncomp)]
paths = ['%s/%s/%s.pvd' % (rootDir, filename, filename) for filename in filenames]

paraview_output_pvd = []
for n in range(ncomp):
    paraview_output_pvd += [PVDReader(registrationName=filenames[n], FileName=paths[n])]
    paraview_output_pvd[-1].CellArrays = ['attribute']
    paraview_output_pvd[-1].PointArrays = ['pres', 'vel']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
paraview_output_pvdDisplay = []
for n in range(ncomp):
    paraview_output_pvdDisplay += [Show(paraview_output_pvd[n], renderView1, 'UnstructuredGridRepresentation')]

    # trace defaults for the display properties.
    paraview_output_pvdDisplay[-1].Representation = 'Surface'
    paraview_output_pvdDisplay[-1].ColorArrayName = [None, '']
    paraview_output_pvdDisplay[-1].SelectTCoordArray = 'None'
    paraview_output_pvdDisplay[-1].SelectNormalArray = 'None'
    paraview_output_pvdDisplay[-1].SelectTangentArray = 'None'
    paraview_output_pvdDisplay[-1].OSPRayScaleArray = 'pres'
    paraview_output_pvdDisplay[-1].OSPRayScaleFunction = 'PiecewiseFunction'
    paraview_output_pvdDisplay[-1].SelectOrientationVectors = 'None'
    paraview_output_pvdDisplay[-1].ScaleFactor = 0.10000000000000014
    paraview_output_pvdDisplay[-1].SelectScaleArray = 'None'
    paraview_output_pvdDisplay[-1].GlyphType = 'Arrow'
    paraview_output_pvdDisplay[-1].GlyphTableIndexArray = 'None'
    paraview_output_pvdDisplay[-1].GaussianRadius = 0.005000000000000007
    paraview_output_pvdDisplay[-1].SetScaleArray = ['POINTS', 'pres']
    paraview_output_pvdDisplay[-1].ScaleTransferFunction = 'PiecewiseFunction'
    paraview_output_pvdDisplay[-1].OpacityArray = ['POINTS', 'pres']
    paraview_output_pvdDisplay[-1].OpacityTransferFunction = 'PiecewiseFunction'
    paraview_output_pvdDisplay[-1].DataAxesGrid = 'GridAxesRepresentation'
    paraview_output_pvdDisplay[-1].PolarAxes = 'PolarAxesRepresentation'
    paraview_output_pvdDisplay[-1].ScalarOpacityUnitDistance = 0.18959736572176963
    paraview_output_pvdDisplay[-1].OpacityArrayName = ['POINTS', 'pres']
    paraview_output_pvdDisplay[-1].SelectInputVectors = [None, '']
    paraview_output_pvdDisplay[-1].WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    paraview_output_pvdDisplay[-1].ScaleTransferFunction.Points = [1.034992516704781, 0.0, 0.5, 0.0, 40.296163205568156, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    paraview_output_pvdDisplay[-1].OpacityTransferFunction.Points = [1.034992516704781, 0.0, 0.5, 0.0, 40.296163205568156, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5000000000000001, 1.4999999999999998, 10000.0]
renderView1.CameraFocalPoint = [0.5000000000000001, 1.4999999999999998, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

for n in range(ncomp):
    # set active source
    SetActiveSource(paraview_output_pvdDisplay[n])

    # set scalar coloring
    ColorBy(paraview_output_pvdDisplay[n], ('POINTS', 'vel', 'Magnitude'))
    # ColorBy(paraview_output_pvdDisplay[n], ('POINTS', 'pres'))

    # rescale color and/or opacity maps used to include current data range
    paraview_output_pvdDisplay[n].RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    paraview_output_pvdDisplay[n].SetScalarBarVisibility(renderView1, False)

# get 2D transfer function for 'vel'
velTF2D = GetTransferFunction2D('vel')

# get color transfer function/color map for 'vel'
velLUT = GetColorTransferFunction('vel')
velLUT.TransferFunction2D = velTF2D
velLUT.RGBPoints = [1.7563093092470768e-06, 0.231373, 0.298039, 0.752941, 0.45570586328035567, 0.865003, 0.865003, 0.865003, 0.911409970251402, 0.705882, 0.0156863, 0.14902]
velLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'vel'
velPWF = GetOpacityTransferFunction('vel')
velPWF.Points = [1.7563093092470768e-06, 0.0, 0.5, 0.0, 0.911409970251402, 1.0, 0.5, 0.0]
velPWF.ScalarRangeInitialized = 1

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1187, 1154)

#-----------------------------------
# saving camera placements for views

# # current camera placement for renderView1
# renderView1.InteractionMode = '2D'
# renderView1.CameraPosition = [1.0728459999485958, 1.0649950687890346, 10000.0]
# renderView1.CameraFocalPoint = [1.0728459999485958, 1.0649950687890346, 0.0]
# renderView1.CameraParallelScale = 1.2526827963856217

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [16.359130599262404, 16.042753498411386, 10000.0]
renderView1.CameraFocalPoint = [16.359130599262404, 16.042753498411386, 0.0]
renderView1.CameraParallelScale = 16.36196422443889

# save screenshot
SaveScreenshot('C:/Users/chung28/Desktop/test.png', renderView1, ImageResolution=[1187, 1154])

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).