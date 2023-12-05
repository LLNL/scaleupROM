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

num_basis = 15
figure_dir = 'C:\\Users\\chung28\\Desktop\\slides\\figures\\stokes-4-comp-channel\\basis'

# create a new 'PVD Reader'
stokes_comp_universal_0pvd = PVDReader(registrationName='stokes_comp_universal_0.pvd', FileName='\\\\wsl$\\Ubuntu-22.04\\home\\kevin\\scaleupROM\\playground\\stokes_channel\\stokes_comp_universal_0\\stokes_comp_universal_0.pvd')
stokes_comp_universal_0pvd.CellArrays = ['attribute']
stokes_comp_universal_0pvd.PointArrays = ['pres_basis_%d' % k for k in range(num_basis)] + ['vel_basis_%d' % k for k in range(num_basis)]
# stokes_comp_universal_0pvd.PointArrays = ['pres_basis_0', 'pres_basis_1', 'pres_basis_10', 'pres_basis_11', 'pres_basis_12', 'pres_basis_13', 'pres_basis_14', 'pres_basis_2', 'pres_basis_3', 'pres_basis_4', 'pres_basis_5', 'pres_basis_6', 'pres_basis_7', 'pres_basis_8', 'pres_basis_9', 'vel_basis_0', 'vel_basis_1', 'vel_basis_10', 'vel_basis_11', 'vel_basis_12', 'vel_basis_13', 'vel_basis_14', 'vel_basis_2', 'vel_basis_3', 'vel_basis_4', 'vel_basis_5', 'vel_basis_6', 'vel_basis_7', 'vel_basis_8', 'vel_basis_9']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
stokes_comp_universal_0pvdDisplay = Show(stokes_comp_universal_0pvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
stokes_comp_universal_0pvdDisplay.Representation = 'Surface'
stokes_comp_universal_0pvdDisplay.ColorArrayName = [None, '']
stokes_comp_universal_0pvdDisplay.SelectTCoordArray = 'None'
stokes_comp_universal_0pvdDisplay.SelectNormalArray = 'None'
stokes_comp_universal_0pvdDisplay.SelectTangentArray = 'None'
stokes_comp_universal_0pvdDisplay.OSPRayScaleArray = 'pres_basis_0'
stokes_comp_universal_0pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
stokes_comp_universal_0pvdDisplay.SelectOrientationVectors = 'None'
stokes_comp_universal_0pvdDisplay.ScaleFactor = 0.10000000000000003
stokes_comp_universal_0pvdDisplay.SelectScaleArray = 'None'
stokes_comp_universal_0pvdDisplay.GlyphType = 'Arrow'
stokes_comp_universal_0pvdDisplay.GlyphTableIndexArray = 'None'
stokes_comp_universal_0pvdDisplay.GaussianRadius = 0.005000000000000001
stokes_comp_universal_0pvdDisplay.SetScaleArray = ['POINTS', 'pres_basis_0']
stokes_comp_universal_0pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
stokes_comp_universal_0pvdDisplay.OpacityArray = ['POINTS', 'pres_basis_0']
stokes_comp_universal_0pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
stokes_comp_universal_0pvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
stokes_comp_universal_0pvdDisplay.PolarAxes = 'PolarAxesRepresentation'
stokes_comp_universal_0pvdDisplay.ScalarOpacityUnitDistance = 0.24955016910866956
stokes_comp_universal_0pvdDisplay.OpacityArrayName = ['POINTS', 'pres_basis_0']
stokes_comp_universal_0pvdDisplay.SelectInputVectors = [None, '']
stokes_comp_universal_0pvdDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
stokes_comp_universal_0pvdDisplay.ScaleTransferFunction.Points = [0.04820203539272803, 0.0, 0.5, 0.0, 0.0551585124580708, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
stokes_comp_universal_0pvdDisplay.OpacityTransferFunction.Points = [0.04820203539272803, 0.0, 0.5, 0.0, 0.0551585124580708, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5000000000000001, 0.5000000000000001, 10000.0]
renderView1.CameraFocalPoint = [0.5000000000000001, 0.5000000000000001, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()
# Adjust camera

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5000000000000001, 0.5000000000000001, 10000.0]
renderView1.CameraFocalPoint = [0.5000000000000001, 0.5000000000000001, 0.0]
renderView1.CameraParallelScale = 0.7071067811865476
# Adjust camera

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5000000000000001, 0.5000000000000001, 10000.0]
renderView1.CameraFocalPoint = [0.5000000000000001, 0.5000000000000001, 0.0]
renderView1.CameraParallelScale = 0.7071067811865476

# set scalar coloring
ColorBy(stokes_comp_universal_0pvdDisplay, ('POINTS', 'pres_basis_0'))

# rescale color and/or opacity maps used to include current data range
stokes_comp_universal_0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
stokes_comp_universal_0pvdDisplay.SetScalarBarVisibility(renderView1, True)

# get 2D transfer function for 'pres_basis_0'
pres_basis_0TF2D = GetTransferFunction2D('pres_basis_0')

# get color transfer function/color map for 'pres_basis_0'
pres_basis_0LUT = GetColorTransferFunction('pres_basis_0')
pres_basis_0LUT.TransferFunction2D = pres_basis_0TF2D
pres_basis_0LUT.RGBPoints = [0.04820203539272803, 0.231373, 0.298039, 0.752941, 0.051680273925399416, 0.865003, 0.865003, 0.865003, 0.0551585124580708, 0.705882, 0.0156863, 0.14902]
pres_basis_0LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'pres_basis_0'
pres_basis_0PWF = GetOpacityTransferFunction('pres_basis_0')
pres_basis_0PWF.Points = [0.04820203539272803, 0.0, 0.5, 0.0, 0.0551585124580708, 1.0, 0.5, 0.0]
pres_basis_0PWF.ScalarRangeInitialized = 1
# Adjust camera

# hide color bar/color legend
stokes_comp_universal_0pvdDisplay.SetScalarBarVisibility(renderView1, False)
# Adjust camera

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5000000000000001, 0.5000000000000001, 10000.0]
renderView1.CameraFocalPoint = [0.5000000000000001, 0.5000000000000001, 0.0]
renderView1.CameraParallelScale = 0.5189178395400392

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(782, 790)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.5000000000000001, 0.5000000000000001, 10000.0]
renderView1.CameraFocalPoint = [0.5000000000000001, 0.5000000000000001, 0.0]
renderView1.CameraParallelScale = 0.5189178395400392

for k in range(num_basis):
    # get 2D transfer function for 'pres_basis_0'
    pres_basis_TF2D = GetTransferFunction2D('pres_basis_%d' % k)

    # get color transfer function/color map for 'pres_basis_0'
    pres_basis_LUT = GetColorTransferFunction('pres_basis_%d' % k)
    pres_basis_LUT.TransferFunction2D = pres_basis_TF2D
    pres_basis_LUT.RGBPoints = [0.04820203539272803, 0.231373, 0.298039, 0.752941, 0.051680273925399416, 0.865003, 0.865003, 0.865003, 0.0551585124580708, 0.705882, 0.0156863, 0.14902]
    pres_basis_LUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'pres_basis_0'
    pres_basis_PWF = GetOpacityTransferFunction('pres_basis_%d' % k)
    pres_basis_PWF.Points = [0.04820203539272803, 0.0, 0.5, 0.0, 0.0551585124580708, 1.0, 0.5, 0.0]
    pres_basis_PWF.ScalarRangeInitialized = 1
    # Adjust camera

    # set scalar coloring
    ColorBy(stokes_comp_universal_0pvdDisplay, ('POINTS', 'pres_basis_%d' % k))
    # rescale color and/or opacity maps used to include current data range
    stokes_comp_universal_0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)
    # hide color bar/color legend
    stokes_comp_universal_0pvdDisplay.SetScalarBarVisibility(renderView1, False)

    # save screenshot
    SaveScreenshot('%s\\p\\%d.png' % (figure_dir, k), renderView1, ImageResolution=[782, 790])

    # get 2D transfer function for 'vel_basis_0'
    vel_basis_TF2D = GetTransferFunction2D('vel_basis_%d' % k)

    # get color transfer function/color map for 'vel_basis_0'
    vel_basis_LUT = GetColorTransferFunction('vel_basis_%d' % k)
    vel_basis_LUT.TransferFunction2D = vel_basis_TF2D
    vel_basis_LUT.RGBPoints = [1.4547756297732489e-08, 0.231373, 0.298039, 0.752941, 0.00024627745173383023, 0.865003, 0.865003, 0.865003, 0.0004925403557113627, 0.705882, 0.0156863, 0.14902]
    vel_basis_LUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'vel_basis_0'
    vel_basis_PWF = GetOpacityTransferFunction('vel_basis_%d' % k)
    vel_basis_PWF.Points = [1.4547756297732489e-08, 0.0, 0.5, 0.0, 0.0004925403557113627, 1.0, 0.5, 0.0]
    vel_basis_PWF.ScalarRangeInitialized = 1

    # set scalar coloring
    ColorBy(stokes_comp_universal_0pvdDisplay, ('POINTS', 'vel_basis_%d' % k, 'X'))
    # rescale color and/or opacity maps used to include current data range
    stokes_comp_universal_0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)
    # show color bar/color legend
    stokes_comp_universal_0pvdDisplay.SetScalarBarVisibility(renderView1, False)

    # save screenshot
    SaveScreenshot('%s\\u\\%d.png' % (figure_dir, k), renderView1, ImageResolution=[782, 790])

    # set scalar coloring
    ColorBy(stokes_comp_universal_0pvdDisplay, ('POINTS', 'vel_basis_%d' % k, 'Y'))
    # rescale color and/or opacity maps used to include current data range
    stokes_comp_universal_0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)
    # show color bar/color legend
    stokes_comp_universal_0pvdDisplay.SetScalarBarVisibility(renderView1, False)

    # save screenshot
    SaveScreenshot('%s\\v\\%d.png' % (figure_dir, k), renderView1, ImageResolution=[782, 790])

    # set scalar coloring
    ColorBy(stokes_comp_universal_0pvdDisplay, ('POINTS', 'vel_basis_%d' % k, 'Magnitude'))
    # rescale color and/or opacity maps used to include current data range
    stokes_comp_universal_0pvdDisplay.RescaleTransferFunctionToDataRange(True, False)
    # show color bar/color legend
    stokes_comp_universal_0pvdDisplay.SetScalarBarVisibility(renderView1, False)

    # save screenshot
    SaveScreenshot('%s\\vel_mag\\%d.png' % (figure_dir, k), renderView1, ImageResolution=[782, 790])
