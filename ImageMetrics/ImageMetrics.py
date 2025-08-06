import os
import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import tempfile
import qt

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLMarkupsPlaneNode, vtkMRMLTableNode
from typing import Annotated, Optional
import numpy as np
import logging
import time
import sys

try:
    import matplotlib
except ModuleNotFoundError:
    slicer.util.pip_install("matplotlib")
    import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a simple handler to output to console
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

#
# ImageMetrics
#


class ImageMetrics(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ImageMetrics")  # TODO: make this more human readable by adding spaces
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Quantification")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ashkan Pakzad (University of Melbourne)"]  
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""Measure reference-less image quality metrics such as signal-to-noise ratio, contrast and resolution. 
                                 https://github.com/ashkanpakzad/SlicerImageMetrics""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""Developed by Ashkan Pakzad (ashkanpakzad.github.io), University of Melbourne (ashkan.pakzad@unimelb.edu.au). 
                                            Enabled by funding from the National Health and Medical Research Council [2021/GNT2011204]
                                            https://impact-mi.sydney.edu.au/ """)

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)


#
# ImageMetricsParameterNode   
#


@parameterNodeWrapper
class ImageMetricsParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to measure image quality.
    annotation - The annotation (plane) to measure image quality.
    contrast - The contrast (plane) to measure image quality.
    table - where to store the results.
    debug - whether to show debug messages and plots.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    annotation: vtkMRMLMarkupsPlaneNode
    contrast: vtkMRMLMarkupsPlaneNode
    table: vtkMRMLTableNode = None
    debug: bool = False


#
# ImageMetricsWidget
#


class ImageMetricsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ImageMetrics.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ImageMetricsLogic()

        # configure markups widget to only allow plane nodes
        # self.ui.annotationWidget.markupsSelectorComboBox().setToolTip("Pick plane to measure image quality.")
        self.ui.annotationWidget.markupsSelectorComboBox().nodeTypes = ["vtkMRMLMarkupsPlaneNode"]
        self.ui.annotationWidget.markupsSelectorComboBox().selectNodeUponCreation = True
        self.ui.annotationWidget.markupsSelectorComboBox().addEnabled = True
        self.ui.annotationWidget.markupsSelectorComboBox().removeEnabled = True
        self.ui.annotationWidget.markupsSelectorComboBox().noneEnabled = False
        self.ui.annotationWidget.markupsSelectorComboBox().showHidden = False
        # configure table widget to not show
        self.ui.annotationWidget.tableWidget().setVisible(False)

        # self.ui.annotationWidget.markupsSelectorComboBox().setToolTip("Pick plane to measure image quality.")
        self.ui.contrastWidget.markupsSelectorComboBox().nodeTypes = ["vtkMRMLMarkupsPlaneNode"]
        self.ui.contrastWidget.markupsSelectorComboBox().selectNodeUponCreation = True
        self.ui.contrastWidget.markupsSelectorComboBox().addEnabled = True
        self.ui.contrastWidget.markupsSelectorComboBox().removeEnabled = True
        self.ui.contrastWidget.markupsSelectorComboBox().noneEnabled = True
        self.ui.contrastWidget.markupsSelectorComboBox().showHidden = False
        # configure table widget to not show
        self.ui.contrastWidget.tableWidget().setVisible(False)

        # Create generic parameter change handlers
        self.onInputChanged = lambda node: self.onParameterChanged('inputVolume', node)
        self.onAnnotationChanged = lambda node: self.onParameterChanged('annotation', node)
        self.onContrastChanged = lambda node: self.onParameterChanged('contrast', node)
        self.onTableChanged = lambda node: self.onParameterChanged('table', node)
        
        # Connect the handlers
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputChanged)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onMarkupChange)
        self.ui.annotationWidget.markupsSelectorComboBox().connect("currentNodeChanged(vtkMRMLNode*)", self.onAnnotationChanged)
        self.ui.annotationWidget.markupsSelectorComboBox().connect("currentNodeChanged(vtkMRMLNode*)", self._onMarkupChange)
        self.ui.annotationWidget.connect("updateFinished()", self._onMarkupChange)
        self.ui.contrastWidget.markupsSelectorComboBox().connect("currentNodeChanged(vtkMRMLNode*)", self.onContrastChanged)
        self.ui.contrastWidget.connect("updateFinished()", self._onMarkupChange)

        self.ui.TableSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onTableChanged)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # debug checkbox
        self.ui.debugBox.connect("toggled(bool)", self.onDebugChanged)

        # connect live update
        self.ui.liveBox.connect("toggled(bool)", self.onLiveChanged)
        self.ui.liveText.connect("textChanged(const QString&)", self._onLiveUpdate)
        self.ui.liveCText.connect("textChanged(const QString&)", self._onLiveUpdate)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # if node already selected, update live textbox
        if self.ui.annotationWidget.markupsSelectorComboBox().currentNode():
            self._onMarkupChange()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            # TODO check if need
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.onMarkupChange)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
        
    def setParameterNode(self, inputParameterNode: Optional[ImageMetricsParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
            # GUI default is registered
            self.onAnnotationChanged(self.ui.annotationWidget.markupsSelectorComboBox().currentNode())

    def _onMarkupChange(self,  caller=None, event=None):
        '''If annotation is changed, update live textbox'''
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.annotation and self.ui.liveBox.isChecked():
            self._onLiveUpdate()

    def _onLiveUpdate(self):
        if self.ui.annotationWidget.markupsSelectorComboBox().currentNode() and \
           self.ui.annotationWidget.markupsSelectorComboBox().currentNode().GetNumberOfControlPoints() > 0:
            try:
                # process with or without contrast node
                contrastNode = self.ui.contrastWidget.markupsSelectorComboBox().currentNode()
                results = self.logic.process(
                    self.ui.inputSelector.currentNode(),
                    self.ui.annotationWidget.markupsSelectorComboBox().currentNode(),
                    contrastNode,
                    None,
                    plotLabel=self.ui.plotLabel
                )
                # select results to show
                showresults1 = {k: v for k, v in results.items() if k in ['shape', 'min', 'max', 'mean', 'std', 'SNR', 'resX', 'resY', 'res2D', 'res2D_L2']}
                showresults2 = {k: v for k, v in results.items() if k in ['cShape', 'visibility', 'CNR']}
                # Format results as a readable string
                result_str1 = "\n".join(
                    f"{k}: {v:.2f}" if isinstance(v, float) and not isinstance(v, bool)
                    else f"{k}: {v:.2e}" if isinstance(v, np.float32)
                    else f"{k}: {v:d}" if isinstance(v, int)
                    else f"{k}: {v}" for k, v in showresults1.items()
                )
                result_str2 = "\n".join(
                    f"{k}: {v:.2f}" if isinstance(v, float) and not isinstance(v, bool)
                    else f"{k}: {v:.2e}" if isinstance(v, np.float32)
                    else f"{k}: {v:d}" if isinstance(v, int)
                    else f"{k}: {v}" for k, v in showresults2.items()
                )
                # update live text
                self.ui.liveText.setText(result_str1)
                self.ui.liveCText.setText(result_str2)
            except Exception as e:
                self.ui.liveText.setText(f'error: {e}')
                self.ui.liveCText.setText(f'error: {e}')
        else:
            self.ui.liveText.setText("No control points on plane.")
            self.ui.liveCText.setText("")

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.annotation:
            self.ui.applyButton.toolTip = _("Compute image quality metrics on annotation")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select source and annotation inputs")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # if no table is selected, create a new one with default name, and select it
            if not self.ui.TableSelector.currentNode():
                tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
                tableNode.SetName("ImageMetricsTable")
                self.ui.TableSelector.setCurrentNode(tableNode)
            # get contrast node if selected
            if self.ui.contrastWidget.markupsSelectorComboBox().currentNode():
                contrastNode = self.ui.contrastWidget.markupsSelectorComboBox().currentNode()
            else:
                contrastNode = None
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.annotationWidget.markupsSelectorComboBox().currentNode(),
                                contrastNode, self.ui.TableSelector.currentNode(), plotLabel=self.ui.plotLabel)
            # show table 
            if self.ui.TableSelector.currentNode():
                slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveTableID(self.ui.TableSelector.currentNode().GetID())
                slicer.app.applicationLogic().PropagateTableSelection()

    def onParameterChanged(self, parameterName, value):
        """
        Generic function to handle parameter changes.
        
        Args:
            parameterName (str): Name of the parameter to update
            value: The new value for the parameter
        """
        if self._parameterNode:
            setattr(self._parameterNode, parameterName, value)
        self._checkCanApply()


    def onDebugChanged(self, value):
        if self._parameterNode:
            self._parameterNode.debug = value
        logger.setLevel(logging.DEBUG if value else logging.INFO)
        # Show debug status in Slicer console
        if value:
            print("DEBUG MODE ENABLED - Check Slicer console for detailed output")
            slicer.util.showStatusMessage("ImageMetrics: Debug mode enabled", 3000)
        else:
            print("DEBUG MODE DISABLED")
            slicer.util.showStatusMessage("ImageMetrics: Debug mode disabled", 3000)

    def onLiveChanged(self, value):
        if self._parameterNode:
            self._parameterNode.live = value
        # Show live status in Slicer console
        if value:
            print("LIVE MODE ENABLED - Check ImageMetrics widget for results")
            slicer.util.showStatusMessage("ImageMetrics: Live mode enabled", 3000)
        else:
            print("LIVE MODE DISABLED")
            slicer.util.showStatusMessage("ImageMetrics: Live mode disabled", 3000)


#
# ImageMetricsLogic
#


class ImageMetricsLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return ImageMetricsParameterNode(super().getParameterNode())

    def process(self,
                sourceNode: vtkMRMLScalarVolumeNode,
                annotationNode: vtkMRMLMarkupsPlaneNode,
                contrastNode: vtkMRMLMarkupsPlaneNode = None,
                tableNode: vtkMRMLTableNode = None,
                plotLabel: qt.QLabel = None
                ) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param sourceNode: volume to measure image quality
        :param annotationNode: annotation (plane) to measure image quality
        :param contrastNode: contrast (plane) to measure image quality
        :param tableNode: table to store the results
        :param plotLabel: label to show the plot
        """
        # validate inputs
        if not sourceNode or not annotationNode:
            raise ValueError("Input or annotation is invalid")
        
        startTime = time.time()
        logger.debug(f"Started for volume: {sourceNode.GetName()}, annotation: {annotationNode.GetName()}")

        row = {
            'volume': sourceNode.GetName(),
        }

        # get volume affine RAS to IJK
        affine = self.getAffine(sourceNode)
        spacings = sourceNode.GetSpacing()

        # get patch IJK
        boundsRAS = self.getPointNormalPlaneBoundsRAS(annotationNode)
        patch, boundsIJK = self.getPatch(sourceNode, affine, boundsRAS)
        row.update({
            'I': boundsIJK[:,0],
            'J': boundsIJK[:,1],
            'K': boundsIJK[:,2],
        })

        # identify relevant spacings for plane
        plane_dim = np.where(np.abs(np.diff(boundsIJK[:,:3], axis=0))<2)[1]
        logger.debug(f"Plane dimension: {plane_dim}")
        spacings2 = np.delete(spacings, plane_dim)

        # get stats
        stats = self.getPatchStats(patch, spacings2)
        row.update(stats)

        # get contrast patch IJK
        if contrastNode:
            boundsRAScontrast = self.getPointNormalPlaneBoundsRAS(contrastNode)
            contrastPatch, boundsIJKcontrast = self.getPatch(sourceNode, affine, boundsRAScontrast)
            row.update({
                'cShape': contrastPatch.shape,
                'cI': boundsIJKcontrast[:,0],
                'cJ': boundsIJKcontrast[:,1],
                'cK': boundsIJKcontrast[:,2],
            })
            # take mean along shorter axis
            short_axis = 0 if contrastPatch.shape[0] < contrastPatch.shape[1] else 1
            logger.debug(f"Contrast patch shape: {contrastPatch.shape}, short axis: {short_axis}")
            contrastProfile = np.mean(contrastPatch, axis=short_axis)
            visibility, bin_edges = self.computeVisibility(contrastProfile)
            row.update({
                'visibility': float(visibility),
                'CNR': visibility * stats['SNR'],
            })
            
            # plot contrast profile
            self.plotVis(contrastProfile, bin_edges, labelWidget=plotLabel)

        # update table
        if tableNode:
            logger.debug("Updating results table")
            self.updateTable(tableNode, row)

        stopTime = time.time()
        logger.debug(f"Completed in {stopTime-startTime:.2f} seconds")
        return row
        
    def getAffine(self, sourceNode: vtkMRMLScalarVolumeNode) -> np.ndarray:
        '''Get transform from source RAS to IJK coordinates'''
        logger.debug("Getting RAS to IJK transform")
        rasToIJK = vtk.vtkMatrix4x4()
        sourceNode.GetRASToIJKMatrix(rasToIJK)
        affine = slicer.util.arrayFromVTKMatrix(rasToIJK)
        logger.debug(f"Affine matrix:\n{affine}\n")
        return affine
    
    def getPointNormalPlaneBoundsRAS(self, annotationNode: vtkMRMLMarkupsPlaneNode) -> np.ndarray:
        '''Get bounds of annotation in RAS coordinates'''
        logger.debug("Getting annotation bounds in RAS coordinates")
        # validate annotation
        if not isinstance(annotationNode, vtkMRMLMarkupsPlaneNode):
            raise ValueError("Annotation is not a plane")
        if annotationNode.PlaneTypePointNormal != 1:
            raise ValueError("Annotation Plane must be of type PointNormal")

        # get RAS bounds of annotation
        boundsRAS = [0.0] * 6
        annotationNode.GetRASBounds(boundsRAS)
        # reshape boundsRAS from [xmin,xmax,ymin,ymax,zmin,zmax] to [[min],[max]]
        boundsRAS = np.array(boundsRAS).reshape(3,2).T
        # add column of ones
        boundsRAS = np.concatenate([boundsRAS, np.ones((2, 1))], axis=1)
        logger.debug(f"Bounds RAS shape: {boundsRAS.shape}")
        return boundsRAS

    def getPatch(self, sourceNode: vtkMRMLScalarVolumeNode, affine: np.ndarray, boundsRAS: np.ndarray) -> np.ndarray:
        '''Get patch from annotation and source'''
        logger.debug("Extracting patch from volume")

        # get source array
        source_array = slicer.util.arrayFromVolume(sourceNode)
        source_array = np.swapaxes(source_array, 0, 2)
        logger.debug(f"Source array shape: {source_array.shape}")

        # transform bounds to IJK coordinates
        boundsIJK = np.dot(affine, boundsRAS.T).T
        logger.debug(f"Bounds in floating point IJK coordinates: {boundsIJK[:,:-1]}")
        # convert to indices
        boundsIJK = (np.round(boundsIJK)).astype(int)
        # sort each column by ascending order
        boundsIJK = np.sort(boundsIJK, axis=0)
        logger.debug(f"Bounds in integer IJK coordinates: {boundsIJK[:,:-1]}")
        # get patch from source array
        patch = source_array[boundsIJK[0,0]:boundsIJK[1,0]+1, boundsIJK[0,1]:boundsIJK[1,1]+1, boundsIJK[0,2]:boundsIJK[1,2]+1]
        patch = patch.squeeze()
        logger.debug(f"Patch shape: {patch.shape}")
        logger.debug(f"Patch value range: [{np.min(patch):.2e}, {np.max(patch):.2e}]")
        return patch, boundsIJK
    
    def getPatchStats(self, patch: np.ndarray, spacings2: np.ndarray) -> dict:
        '''Get stats from patch'''
        logger.debug("Computing patch statistics")
        if patch.size == 0:
            raise ValueError("Patch is empty")
        SNR = float(np.mean(patch) / np.std(patch))
        logger.debug(f"SNR calculated: {SNR:.3f}")
        # resolution using power spectrum
        res = self.computeResolution(patch.T, spacings2)
        stats = {
            'shape': patch.shape,
            'min': np.min(patch),
            'max': np.max(patch),
            'mean': np.mean(patch),
            'std': np.std(patch),
            'SNR': SNR,
        }
        stats.update(res)
        return stats
    
    def computeVisibility(self, contrastProfile: np.ndarray) -> tuple[float, np.ndarray]:
        '''Compute visibility of contrast profile'''
        logger.debug(f"Computing visibility")

        # get histogram of contrast profile
        num_bins = 5
        _, bin_edges = np.histogram(contrastProfile, bins=num_bins)

        # get values in each bin
        # Preallocate bins list with empty arrays
        bins = [np.array([]) for _ in range(num_bins)]
        for i in range(num_bins):
            bins[i] = contrastProfile[(contrastProfile >= bin_edges[i]) & (contrastProfile < bin_edges[i+1])]
        
        # get mean of lower and upper bins
        lower_mean = np.mean(bins[0]) if len(bins[0]) > 0 else 0
        upper_mean = np.mean(bins[-1]) if len(bins[-1]) > 0 else 0
        logger.debug(f"bin means: {lower_mean}, {upper_mean}")
        if lower_mean < 0 and upper_mean > 0:
            visibility = -1.0
            logger.info("Visibility cannot be calculated, because the lower bin mean is negative and the upper bin mean is positive")
        elif lower_mean == 0 and upper_mean == 0:
            visibility = 0.0
            logger.info("Visibility is zero, because the lower and upper bin means are zero")
        else:
            visibility = (upper_mean - lower_mean) / (upper_mean + lower_mean)
        logger.debug(f"visibility: {visibility}")
        return visibility, bin_edges

    def computeResolution(self, patch_in: np.ndarray, spacings2: np.ndarray, method: str = 'DELTAX') -> dict:
        '''Compute spatial resolution using FFT and profile analysis'''
        # validate inputs
        if patch_in.size == 0:
            raise ValueError("Patch is empty")
        if len(spacings2) != 2:
            raise ValueError("Expected 2 values for spacings")
        if method not in ['FWHM', 'DELTA2X', 'DELTAX']:
            raise ValueError(f"Invalid method: {method}")

        logger.debug(f"computeResolution: spacings2={spacings2}")
        logger.debug(f"computeResolution: Input patch shape {patch_in.shape}, method={method}")
        
        # trim to power of 2
        nx, ny = patch_in.shape
        nx_new = int(2**np.floor(np.log2(nx)))  # Use floor instead of ceil to avoid expanding
        ny_new = int(2**np.floor(np.log2(ny)))
        # Center the trimmed region
        x_start = (nx - nx_new) // 2
        y_start = (ny - ny_new) // 2
        patch = patch_in[x_start:x_start+nx_new, y_start:y_start+ny_new]
        logger.debug(f"computeResolution: centered and trimmed patch shape {patch.shape}")

        # Normalize array
        l1_norm = np.sum(np.abs(patch))
        patch_norm = patch / l1_norm
        logger.debug(f"computeResolution: L1 norm = {l1_norm:.6f}")

        # TODO in future average and use 1D FFTs?
        # Take FFT and get magnitude
        fft = np.fft.fft2(patch_norm)
        # shift high frequencies to center
        fft = np.fft.fftshift(fft)
        fft_mag = np.abs(fft)
        logger.debug(f"computeResolution: FFT shape {fft_mag.shape}, max magnitude = {np.max(fft_mag):.6f}")
        self.debug2Dplot(fft, "debug: FFT after shift")

        # remove central peak
        center_x, center_y = fft_mag.shape[0]//2, fft_mag.shape[1]//2
        fft_mag[center_x, center_y] = 0

        # Get profiles by averaging in each dimension
        profile_x = np.mean(fft_mag, axis=0)
        profile_y = np.mean(fft_mag, axis=1)
        logger.debug(f"computeResolution: Profile X shape {profile_x.shape}, Y shape {profile_y.shape}")
        self.debug1Dplot(profile_x, "Profile X")
        self.debug1Dplot(profile_y, "Profile Y")

        # Get 1D sigma_fft and FWHM bounds of peak 
        sigma1X_fft, fit_x, left_fwhm_x, right_fwhm_x = self._compute_sigma_fft_1d(profile_x, method)
        sigma1Y_fft, fit_y, left_fwhm_y, right_fwhm_y = self._compute_sigma_fft_1d(profile_y, method)
        logger.debug(f"computeResolution: 1D resolutions - dx={sigma1X_fft:.3f} (fit={fit_x}), dy={sigma1Y_fft:.3f} (fit={fit_y})")
        
        # Calculate 2D sigma_fft
        sigma2_fft, sigma2L2_fft = self._compute_sigma_2d(fft_mag, left_fwhm_x, right_fwhm_x, left_fwhm_y, right_fwhm_y)
        logger.debug(f"computeResolution: 2D resolutions - sigma2d={sigma2_fft:.3f}, sigmaL2_2d={sigma2L2_fft:.3f}")
        
        # compute 1D resolution [2*sigma] from sigma_fft
        rDx0 = self._compute_resolution_1d(profile_x, sigma1X_fft, method)
        rDy0 = self._compute_resolution_1d(profile_y, sigma1Y_fft, method)
        
        # compute 2D resolution [2*sigma] from sigma_fft
        rD0, rD02 = self._compute_resolution_2d(patch, sigma2_fft, sigma2L2_fft)

        # convert spacings to mm and store
        logger.debug(f"In pixels, resX: {rDx0}, resY: {rDy0}, res2D: {rD0}, res2D_L2: {rD02}\n Spacing: {spacings2}\n In mm, resX: {rDx0 * spacings2[0]}, resY: {rDy0 * spacings2[1]}, res2D: {rD0 * np.sqrt(spacings2[0] * spacings2[1])}, res2D_L2: {rD02 * np.sqrt(spacings2[0] * spacings2[1])}")
        result = {
            'resX': rDx0 * spacings2[0],
            'resY': rDy0 * spacings2[1],
            'res2D': rD0 * np.sqrt(spacings2[0] * spacings2[1]),
            'res2D_L2': rD02 * np.sqrt(spacings2[0] * spacings2[1]),
            'fit_x': fit_x,
            'fit_y': fit_y,
        }
        logger.debug(f"computeResolution: Final result = {result}")
        return result

    def _compute_sigma_fft_1d(self, profile: np.ndarray, method: str = 'FWHM') -> tuple[float, bool, int, int]:
        '''Analyse 1D FFT profile'''
        n = len(profile)
        if n < 4:
            logger.debug(f"_compute_sigma_1d: Profile too short (n={n}), returning default")
            return 1.0, False, -1, -1
        center = n//2

        # Find background level from first 5% and last 5% of profile
        edge_width = max(n//20, 1)
        bg_level = np.mean(np.concatenate([profile[:edge_width], profile[-edge_width:]]))
        logger.debug(f"_compute_sigma_1d: n={n}, center={center}, bg_level={bg_level:.6f}")

        # Find peaks on either side of center
        left_peak = np.argmax(profile[:center-1])
        right_peak = center + 1 + np.argmax(profile[center+1:])
        logger.debug(f"_compute_sigma_1d: Peaks at left={left_peak}, right={right_peak}")
            
        # Find FWHM peaks
        left_height = max(profile[left_peak]/2, bg_level)
        right_height = max(profile[right_peak]/2, bg_level)
        logger.debug(f"_compute_sigma_1d: Half-max heights - left={left_height:.6f}, right={right_height:.6f}")
        
        # determine FWHM 
        left_fwhm = left_peak
        while left_fwhm >= 0 and profile[left_fwhm] > left_height:
            left_fwhm -= 1
                
        right_fwhm = right_peak  
        while right_fwhm < n and profile[right_fwhm] > right_height:
            right_fwhm += 1
            
        logger.debug(f"_compute_sigma_1d: FWHM bounds - left={left_fwhm}, right={right_fwhm}")
            
        if left_fwhm < 0 or right_fwhm >= n or left_fwhm >= right_fwhm:
            logger.debug(f"_compute_sigma_1d: Invalid FWHM bounds, returning default")
            return 1.0, False, -1, -1
            
        if method == 'FWHM':
            sigma1D_fft = (right_fwhm - left_fwhm)/2.35482
            logger.debug(f"_compute_sigma_1d: FWHM method - fwhm={right_fwhm - left_fwhm}, sigma={sigma1D_fft:.6f}")

        elif method == 'DELTA2X':
            # calculate d2kx norm of the central peak
            left_range = max(0, 2 * left_fwhm - center)
            right_range = min(n, 2 * right_fwhm - center + 1)
            logger.debug(f"_compute_sigma_1d: DELTA2X method - range [{left_range}, {right_range}]")
            region = profile[left_range:right_range]

            # Calculate L1 and L2 norms
            L1 = np.sum(region)
            L2 = np.sum(region**2)
            logger.debug(f"_compute_sigma_1d: DELTA2X - L1={L1:.6f}, L2={L2:.6f}")
            
            sigma1D_fft = 0.5 * L1**2 / L2
            logger.debug(f"_compute_sigma_1d: DELTA2X - sigma={sigma1D_fft:.6f}")

        elif method == 'DELTAX':
            # calculate dkx norm of the central peak

            # get region around peaks
            left_range = max(0, 2 * left_fwhm - center)
            right_range = min(n, 2 * right_fwhm - center + 1)
            region = profile[left_range:right_range]
            logger.debug(f"_compute_sigma_1d: DELTAX method - range [{left_range}, {right_range}]")

            # calculate L1 and variance
            L1 = np.sum(region)
            center_offset = np.arange(left_range, right_range) - center
            Var = np.sum(region * center_offset**2)
            sigma1D_fft = np.sqrt(Var/L1)
            logger.debug(f"_compute_sigma_1d: DELTAX - Var={Var:.6f}, L1={L1:.6f}, sigma1D_fft={sigma1D_fft:.6f}")

        else:
            raise ValueError(f"Invalid method: {method}")
            
        return sigma1D_fft, True, left_fwhm, right_fwhm
    
    def _compute_sigma_2d(self, fft_magnitude: np.ndarray, left_fwhm_x: int, right_fwhm_x: int, left_fwhm_y: int, right_fwhm_y: int) -> tuple[float, float]:
        """Calculate 2D resolution from FFT magnitude."""
        nx, ny = fft_magnitude.shape
        center_x, center_y = nx // 2, ny // 2
        logger.debug(f"_compute_sigma_2d: Input shape ({nx}, {ny}), center ({center_x}, {center_y})")
    
        # Check if FWHM bounds are valid (C++ style validation)
        if left_fwhm_x < 0 or right_fwhm_x < 0 or left_fwhm_y < 0 or right_fwhm_y < 0:
            logger.debug(f"_compute_sigma_2d: Invalid FWHM bounds, returning defaults")
            return 1.0, 1.0
            
        if left_fwhm_x >= right_fwhm_x or left_fwhm_y >= right_fwhm_y:
            logger.debug(f"_compute_sigma_2d: Invalid FWHM bounds order, returning defaults")
            return 1.0, 1.0
        
        # Define the region bounds  
        i_start = max(0, left_fwhm_x - (center_x - left_fwhm_x))
        i_end = min(nx, 1 + right_fwhm_x + (right_fwhm_x - center_x))
        j_start = max(0, left_fwhm_y - (center_y - left_fwhm_y))
        j_end = min(ny, 1 + right_fwhm_y + (right_fwhm_y - center_y))
        logger.debug(f"_compute_sigma_2d: ROI bounds - i[{i_start}, {i_end}], j[{j_start}, {j_end}]")
        
        # Extract the region of interest
        region = fft_magnitude[i_start:i_end, j_start:j_end]
        logger.debug(f"_compute_sigma_2d: Region shape {region.shape}")
        
        if region.size == 0:
            logger.debug(f"_compute_sigma_2d: Empty region, returning defaults")
            return 1.0, 1.0
        
        # Create coordinate grids for variance calculation (relative to center)
        i_coords, j_coords = np.meshgrid(
            np.arange(i_start, i_end) - center_x,
            np.arange(j_start, j_end) - center_y,
            indexing='ij'
        )

        # compute L1,L2,Var
        L1 = np.sum(region)
        L2 = np.sum(region**2)
        Var = np.sum(region * (i_coords**2 + j_coords**2))
        # compute 2D sigma and sigmaL2
        sigma = np.sqrt(0.5*Var/L1) # sigma-fft; 0.5 is because the variance of the 2D Gaussian is 2 * sigma^2
        sigmaL2 = 0.5 * np.sqrt(L1**2/L2) # d2k / 2
        
        return sigma, sigmaL2

    def _compute_resolution_1d(self, profile: np.ndarray, sigma1D_fft: float, method: str) -> float:
        '''Compute 1D resolution along dimension t from sigma'''
        if sigma1D_fft > 0:
            if method == 'FWHM' or method == 'DELTAX':
                rDt = 2.0 / (2.0 * sigma1D_fft * np.pi) * len(profile) # for FWHM and sigma-fft, here: resx = 2 * sigmaX = 2 / [2 * PI * sigmaX_fft)]
            elif method == 'DELTA2X':
                rDt = 1.0 / (2.0 * sigma1D_fft) * len(profile) * 1.3 # for d2k
            else:
                raise ValueError(f"Invalid method: {method}")
        rDt0 = rDt if rDt > 1 else 1.0
        logger.debug(f"_compute_resolution_1d: rDt={rDt:.3f}, rDt0={rDt0:.3f}")
        return rDt0

    def _compute_resolution_2d(self, patch: np.ndarray, sigma2_fft: float, sigma2L2_fft: float) -> tuple[float, float]:
        '''Compute 2D resolution from sigma'''
        sqrt_shape = np.sqrt(patch.shape[0] * patch.shape[1])
        if sigma2_fft > 0:
            rD = 2.0 / (2.0 * sigma2_fft * np.pi) * sqrt_shape
        rD0 = rD if rD > 1 else 1.0
        logger.debug(f"_compute_resolution_2d: rD={rD:.3f}, rD0={rD0:.3f}")

        if sigma2L2_fft > 0:
            rD2 = 1.0 / (2.0 * sigma2L2_fft) * sqrt_shape
        rD02 = rD2 if rD2 > 1 else 1.0
        logger.debug(f"_compute_resolution_2d: rD2={rD2:.3f}, rD02={rD02:.3f}")
        return rD0, rD02

    def updateTable(self, tableNode: vtkMRMLTableNode, row: dict) -> None:
        '''Update table with new row'''
        logger.debug(f"Updating table with {len(row)} columns")
        # Create table if empty
        if tableNode.GetNumberOfColumns() == 0:
            logger.debug("Creating new table columns")
            for colName in row.keys():
                col = tableNode.AddColumn()
                col.SetName(colName)
        
        # Add new row
        rowIndex = tableNode.AddEmptyRow()
        logger.debug(f"Added row at index {rowIndex}")
        for colName, value in row.items():
            tableNode.SetCellText(rowIndex, tableNode.GetColumnIndex(colName), str(value))

    def debug2Dplot(self, image2d: np.ndarray, title: str) -> None:
        '''Plot 2D image'''
        # create node
        if logger.getEffectiveLevel() == logging.DEBUG:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "debug_plot")
            slicer.util.updateVolumeFromArray(node, image2d)
            node.SetName(title)

    def debug1Dplot(self,profile: np.ndarray, title: str) -> None:
        '''Plot 1D profile'''
        if logger.getEffectiveLevel() == logging.DEBUG:
            # Create chart node and table node
            chartNode = slicer.util.plot([profile, np.arange(len(profile))], xColumnIndex = 1)
            chartNode.SetTitle(title)

    def plotVis(self, profile: np.ndarray, bin_edges: np.ndarray, labelWidget=None) -> None:
        '''Plot visibility to labelWidget'''
        plt.figure(figsize=(4, 2))
        plt.plot(profile)

        # Draw horizontal lines for middle bin edges
        for edge in bin_edges[1:-1]:
            plt.axhline(y=edge, color='r', linestyle='--', linewidth=1)

        # create a temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, f"plot.png")
        plt.savefig(temp_file)
        plt.close()
        logger.debug(f"saved plot to {temp_file}")

        # Static image view
        if labelWidget is not None:
            pm = qt.QPixmap(temp_file)
            display_width, display_height = 400, 200  # 4:3 aspect ratio
            scaled_pm = pm.scaled(display_width, display_height, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
            labelWidget.setPixmap(scaled_pm)
            labelWidget.setMinimumSize(display_width, display_height)
            labelWidget.setMaximumSize(display_width, display_height)
            labelWidget.setScaledContents(True)

#   
# ImageMetricsTest
#

class ImageMetricsTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()
        # generate test cases
        
        # Generate random noise volume
        imageSize = [256, 256, 256]
        noise = np.random.normal(loc=100, scale=20, size=imageSize)
        self.testVolume1 = slicer.util.addVolumeFromArray(noise.astype(np.float32), name="TestVolume")
        slicer.util.setSliceViewerLayers(background=self.testVolume1, fit=True, rotateToVolumePlane=True)
        self.delayDisplay("Generated test volume with gaussian noise: mean=100, std=20")

        # create a plane measurement in center of test volume
        self.testPlane1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode")
        self.testPlane1.SetName("testPlane")
        self.testPlane1.SetPlaneType(slicer.vtkMRMLMarkupsPlaneNode.PlaneTypePointNormal)
        self.testPlane1.SetCenter(128, 128, 128)
        self.testPlane1.SetNormal(0, 0, 1)
        self.testPlane1.SetSize(86, 86)
        self.delayDisplay("Created test plane node in center of test volume")

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()

        # basic logic process
        self.test_ImageMetricsBasic()
        # TODO: self.test_ImageMetricsTable() - table output functionality
        # TODO: self.test_ImageMetricsContrast() - contrast measurement with two planes

        # different volume cases
        # TODO: self.test_ImageMetricsNonFloat() - non-float volume

        # different plane cases
        # TODO: self.test_ImageMetricsOutOfBounds() - behavior when plane is outside volume bounds
        # TODO: self.test_ImageMetricsNegativeValues() - negative intensity values
        # TODO: self.test_ImageMetricsEdgeCases() - behavior at volume boundaries
        # TODO: self.test_ImageMetricsEmptyPlane() - behavior with zero-sized plane

        # table cases
        # TODO: self.test_ImageMetricsTableColumns() - input table has different columns than expected
        # TODO: self.test_ImageMetricsLiveUpdate() - live update functionality

        # logic
        # TODO: self.test_ImageMetricsComputeContrast() - contrast measurement logic
        # TODO: self.test_ImageMetricsComputeResolution() - resolution measurement logic

        # plotting
        # TODO: self.test_ImageMetricsPlotting() - plot generation and display


    def test_ImageMetricsBasic(self):
        '''Run ImageMetrics'''
        logic = ImageMetricsLogic()
        row = logic.process(self.testVolume1, self.testPlane1)
        
        # Deterministic checks
        self.assertEqual(row['shape'], (86+1, 86+1)) # patch is inclusive of edges
        
        # Stochastic checks
        # Check mean is approximately 100 (within 20% since noise is random)
        self.assertGreater(row['mean'], 80.0)
        self.assertLess(row['mean'], 120.0)
        # check std is approximately 20 (within 20% since noise is random)
        self.assertGreater(row['std'], 16.0)
        self.assertLess(row['std'], 24.0)
        # Check SNR is approximately 5 (within 20% since noise is random)
        self.assertGreater(row['SNR'], 4.0)
        self.assertLess(row['SNR'], 6.0)
        self.delayDisplay("Test_ImageMetricsBasic passed")

    def test_ImageMetrics1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("ImageMetrics1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = ImageMetricsLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
