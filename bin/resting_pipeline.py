#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

"""
 This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numpy.ma
import nibabel
from nipype.interfaces import afni
from nipype.algorithms import confounds
import os, sys, subprocess
import string, random
import re
import networkx as nx
import logging
import math
import argparse
from argparse import ArgumentParser
from nilearn import plotting
from nilearn.image.image import mean_img
from nilearn.connectome import ConnectivityMeasure
import matplotlib as plt
import shutil
import ants

logging.basicConfig(format='%(asctime)s %(message)s ', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def runproc(string):
     logging.info('running: ' + string)
     subprocess.Popen(string,shell=True).wait()

def afile(string):
    if not os.path.isfile(string):
        msg = "%r not found." % string
        raise argparse.ArgumentTypeError(msg)
    return string

description ="""
resting_pipeline.py --func /path/to/run4.bxh --steps all --outpath /here/ -p func
Program to run through Nan-kuei Chen's resting state analysis pipeline:
    steps:
    0 - Convert data to nii in LAS orientation ( we suggest LAS if you are skipping this step )
    1 - Throwaway of initial volumes, then slice time correction
    2 - Motion correction, then regress out motion parameter
    3 - Skull stripping
    4 - Normalization
    5 - Tissue segmentation
    6 - Nuissance regression
    7 - Detrending
    8 - Bandpass filtering
    9 - FWHM smoothing
    10 - Parcellation and produce correlation matrix from label file
      * or split it up:
         10a - Parcellation from label file
         10b - Produce correlation matrix [--func option is ignored if step 10b
              is run by itself unless --dvarsthreshold is specified, and
              --corrts overrides default location for input parcellation
              results (outputpath/corrlabel_ts.txt)]
    11 - Functional connectivity density mapping
"""

parser = ArgumentParser(description=description)
parser.add_argument("-f", "--func",  action="store", type=afile, dest="funcfile",help="Bxh ( or nifti ) file for functional run", metavar="/path/to/BXH")
parser.add_argument("--throwaway",  action="store", type=int, dest="throwaway",help="number of timepoints to dis-regard from beginning of run", metavar="4")
parser.add_argument("--t1",  action="store", type=afile, dest="anatfile",help="bxh ( or nifti ) file for the anatomical T1", metavar="/path/to/BXH")
parser.add_argument("-p", "--prefix",  action="store", type=str, dest="prefix",help="Prefix for all resulting images, defaults to name of input", metavar="func")
parser.add_argument("-s", "--steps",  action="store", type=str, dest="steps",help="Comma seperated string of steps. 'all' will run everything. Default is all", metavar="0,1,2,3", default='all')
parser.add_argument("-o","--outpath",  action="store",type=str, dest="outpath",help="Location to store output files.", metavar="PATH", default='PWD')
parser.add_argument("--sliceorder",  action="store",type=str, choices=['odd', 'up', 'even', 'down'], dest="sliceorder",help="sliceorder if slicetime correction ( odd=interleaved (1,3,5,2,4,6), up=ascending, down=descending, even=interleaved (2,4,6,1,3,5) ).  Default is to read this from input image, if available.", metavar="string")
parser.add_argument("--tr",  action="store", type=float, dest="tr_ms",help="TR of functional data in msec.", metavar="MSEC")
parser.add_argument("--ssref",  action="store", type=afile, dest="sstemplate",help="Pointer to skull stripped template reference image if not using standard brain.", metavar="FILE")
parser.add_argument("--ref",  action="store", type=afile, dest="template",help="Pointer to skull on template reference image if not using standard brain.", metavar="FILE")
parser.add_argument("--gmmask",  action="store", type=afile, dest="gmmask",help="Pointer to GM mask of reference image if not using standard brain.", metavar="FILE")
parser.add_argument("--wmmask",  action="store", type=afile, dest="csfmask",help="Pointer to WM mask of reference image if not using standard brain.", metavar="FILE")
parser.add_argument("--csfmask",  action="store", type=afile, dest="wmmask",help="Pointer to CSF mask of reference image if not using standard brain.", metavar="FILE")
parser.add_argument("--segmenttransform",  action="store", type=afile, dest="segmenttransform",help="Pointer to a transformation file from T1 to BOLD or Template for the segmentation step.", metavar="FILE")
parser.add_argument("--refbrainmask",  action="store", type=afile, dest="refbrainmask",help="Pointer to brain mask of reference image if not using standard brain.", metavar="FILE")
parser.add_argument("--fnirtbrainmask",  action="store", type=afile, dest="fnirtbrainmask",help="Pointer to a brain mask of reference image if not using standard brain for fnirt.", metavar="FILE")
parser.add_argument("--fnirtconfig",  action="store", type=afile, dest="fnirtconfig",help="Pointer to a configuration file for fnirt.", metavar="FILE")
parser.add_argument("--mcparams",  action="store", type=afile, dest="mcparams",help="Pointer to motion parameters file in .par file type.", metavar="FILE.par")
parser.add_argument("--fwhm",  action="store", type=int, dest="fwhm",help="FWHM kernel smoothing in mm (default is 5)", metavar="5", default='5')
parser.add_argument("--refacpoint",  action="store", type=str, dest="refac",help="AC point of reference image if not using standard MNI brain", metavar="45,63,36", default="45,63,36")
parser.add_argument("--flirtcost",  action="store", type=str, choices=['mutualinfo', 'corrratio', 'normmi', 'bbr'], dest="flirtcost",help="Cost function for flirt registration of BOLD images. Default is 'corratio'.", metavar="cost function", default='corratio')
parser.add_argument("--skullstrip",  action="store",type=str, choices=['bet', 'afni', 'ants'], dest="skullstrip",help="Use FSL's BET, AFNI's 3dSkullStrip+3dAutomask or ANTs' for skull stripping data. Default is 'bet'.", metavar="bet/afni", default='bet')
parser.add_argument("--resample-t1",  action="store",type=str, choices=['yes', 'no'], dest="resamplet1",help="Indicates whether to resample T1 to template resolution before registration. This reduces time and memory need considerably. Default is 'yes'.", metavar="yes/no", default='yes')
parser.add_argument("--fval",  action="store", type=float, choices=range(0,1), dest="fval",help="fractional intensity threshold value to use while skull stripping bold. BET default is 0.4 [0:1]. 3dAutoMask default is 0.5 [0.1:0.9]. A lower value makes the mask larger.", metavar="0.4")
parser.add_argument("--anatfval",  action="store", type=float, dest="anatfval",help="fractional intensity threshold value to use while skull stripping ANAT. BET default is 0.5 [0:1]. 3dSkullStrip default is 0.6 [0:1]. A lower value makes the mask larger.", metavar="0.5")
parser.add_argument("--regmethod",  action="store",type=str, choices=['ants', 'fsl'], dest="regmethod",help="Register and normalize images using 'ants' or 'fsl'. Default is 'fsl'.", metavar="ants/fsl", default='fsl')
parser.add_argument("--regressors", action="store", dest="regressors", type=str, choices=['wm', 'csf', 'motion', 'compcor'], nargs='*', help="List nuissance regressors separated by a space. Default is motion, wm amd csf. If CompCor is specified, csf/wm signals will be regressed as part of CompCor.", metavar="regressors", default=['motion', 'wm', 'csf'])
parser.add_argument("--compcor-components",  action="store", type=int, choices=range(0,10), dest="compcor_components",help="Number of principal components to be used by CompCor.", metavar="5", default='5')
parser.add_argument("--detrend",  action="store", type=int, choices=range(0,4), dest="detrend",help="Polynomial up to which to detrend signal. Default is 2 (constant + linear + quadratic).", metavar="2", default='2')
parser.add_argument("--lpfreq",  action="store", type=float, dest="lpfreq",help="Frequency cutoff for lowpass filtering in HZ.  default is .08 Hz", metavar="0.08", default='0.08')
parser.add_argument("--hpfreq",  action="store", type=float, dest="hpfreq",help="Frequency cutoff for highpass filtering in HZ.  default is .01 Hz", metavar="0.01", default='0.01')
parser.add_argument("--corrlabel",  action="store", type=afile, dest="corrlabel",help="Pointer to 3D label containing ROIs for the correlation search. Default is the 116 region AAL label file.", metavar="FILE")
parser.add_argument("--corrtext",  action="store", type=afile, dest="corrtext",help="Pointer to text file containing names/indices for ROIs for the correlation search. Default is the 116 region AAL label txt file.", metavar="FILE")
parser.add_argument("--corrts",  action="store", type=str, dest="corrts",help="If using step 10b by itself, this is the path to parcellation output (default is to use OUTPATH/corrlabel_ts.txt), which will be used as input to the correlation.", metavar="FILE")
parser.add_argument("--dvarsthreshold",  action="store", type=str, dest="dvarsthreshold",help="If specified, this reprsents a DVARS threshold either in BOLD units, or if ending in a '%' character, as a percentage of mean global signal intensity (over the brain mask).  Any volume contributing to a DVARS value greater than this threshold will be excluded (\"scrubbed\") from the (final) correlation step.  DVARS calculation is performed on the results of the last pre-processing step, and is calculated as described by Power, J.D., et al., \"Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion\", NeuroImage(2011).  Note: data is only excluded during the final correlation, and so will never affect any operations that require the full signal, like regression, etc.", metavar="THRESH")
parser.add_argument("--dvarsnumneighbors",  action="store", type=int, dest="dvarsnumneighbors",help="If --dvarsthreshold is specified, then --dvarsnumnumneighbors specifies how many neighboring volumes, before and after the initially excluded volumes, should also be excluded.  Default is 0.", metavar="NUMNEIGHBORS")
parser.add_argument("--fdthreshold",  action="store", type=float, dest="fdthreshold",help="If specified, this reprsents a FD threshold in mm.  Any volume contributing to a FD value greater than this threshold will be excluded (\"scrubbed\") from the (final) correlation step.  FD calculation is performed on the results of the last pre-processing step, and is calculated as described by Power, J.D., et al., \"Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion\", NeuroImage(2011).  Note: data is only excluded during the final correlation, and so will never affect any operations that require the full signal, like regression, etc.", metavar="THRESH")
parser.add_argument("--fdnumneighbors",  action="store", type=int, dest="fdnumneighbors",help="If --fdthreshold is specified, then --fdnumnumneighbors specifies how many neighboring volumes, before and after the initially excluded volumes, should also be excluded.  Default is 0.", metavar="NUMNEIGHBORS")
parser.add_argument("--motionthreshold",  action="store", type=float, dest="motionthreshold",help="If specified, any volume whose motion parameters indicate a movement greater than this threshold (in mm) will be excluded (\"scrubbed\") from the (final) correlation step.  Volume-to-volume movement is calculated per pair of neighboring volumes from the three rotational and three translational parameters generated by mcflirt.  Motion for a pair of neighboring volumes is calculated as the maximum displacement (due to the combined rotation and translation) of any voxel on the 50mm-radius sphere surrounding the center of rotation.  Note: data is only excluded during the final correlation, and so will never affect any operations that require the full signal, like regression, etc.", metavar="THRESH")
parser.add_argument("--motionnumneighbors",  action="store", type=int, dest="motionnumneighbors",help="If --motionthreshold is specified, then --motionnumnumneighbors specifies how many neighboring volumes, before and after the initially excluded volumes, should also be excluded.  Default is 1.", metavar="NUMNEIGHBORS")
parser.add_argument("--motionpar",  action="store", type=str, dest="motionpar",help="If --motionthreshold is specified, then --motionpar specifies the .par file from which the motion parameters are extracted.  If you allow this script to perform motion correction, then this option is ignored.", metavar="FILE.par")
parser.add_argument("--scrubop",  action="store", type=str, choices=['and', 'or'], dest="scrubop", help="If --motionthreshold, --dvarsthreshold, or --fdthreshold are specified, then --scrubop specifies the aggregation operator used to determine the final list of excluded volumes.  Default is 'or', which means a volume will be excluded if *any* of its thresholds are exceeded, whereas 'and' means all the thresholds must be exceeded to be excluded.")
parser.add_argument("--powerscrub", action="store_true", dest="powerscrub", help="Equivalent to specifying --fdthreshold=0.5 --fdnumneighbors=0 --dvarsthreshold=0.5% --dvarsnumneigbhors=0 --scrubop='and', to mimic the method used in the Power et al. article.  Any conflicting options specified before or after this will override these.", default=False)
parser.add_argument("--scrubkeepminvols",  action="store", type=int, dest="scrubkeepminvols",help="If --motionthreshold, --dvarsthreshold, or --fdthreshold are specified, then --scrubminvols specifies the minimum number of volumes that should pass the threshold before doing any correlation.  If the minimum is not met, then the script exits with an error.  Default is to have no minimum.", metavar="NUMVOLS")
parser.add_argument("--fcdmthresh",  action="store", type=float, dest="fcdmthresh",help="R-value threshold to be used in functional connectivity density mapping ( step11 ). Default is set to 0.6. Algorithm from Tomasi et al, PNAS(2010), vol. 107, no. 21. Calculates the fcdm of functional data from last completed step, inside a dilated gray matter mask", metavar="THRESH", default=0.6)
parser.add_argument("--space",  action="store",type=str, choices=['BOLD', 'T1', 'Template'], dest="space",help="Calculate derivatives in 'BOLD', 'T1' or 'Template' space. Default is Template.", metavar="choice of space", default='Template')
parser.add_argument("--cleanup",  action="store_true", dest="cleanup",help="delete files from intermediate steps?")

options = parser.parse_args()
if '-h' in sys.argv:
    parser.print_help()
    raise SystemExit()
    
if '-help' in sys.argv:
    print("Try --help")
    raise SystemExit()

class RestPipe:
    def __init__(self):
        self.initialize()
        for i in self.steps:
            logging.info('starting step' + i)
            if i == '0':
                self.step0()
            elif i == '1':
                self.step1()
            elif i == '2':
                self.step2()
            elif i == '3':
                self.step3()
            elif i == '4':
                self.step4()
            elif i == '5':
                self.step5()
            elif i == '6':
                self.step6()
            elif i == '7':
                self.step7()
            elif i == '8':
                self.step8()
            elif i == '9':
                self.step9()
            elif i == '10':
                self.step10()
            elif i == '10a':
                self.step10a()
            elif i == '10b':
                self.step10b()
            elif i == '11':
                self.step11()

        if options.cleanup == True:
            self.cleanup()


    def initialize(self):
         #if all was defined, set those steps
        if (options.steps == 'all'):
            self.steps = ['0','1','2','3','4','5','6','7','8','9','10','11']
        else:
            #convert unicode str, push into obj
            self.steps = options.steps.split(',')
            for i in range(len(self.steps)):
                self.steps[i] = str(self.steps[i])

        self.needfunc = True
        if len(self.steps) == 1 and '10b' in self.steps:
            self.needfunc = False

        #make output directory
        if (options.outpath == 'PWD'):
            self.outpath = os.environ['PWD']
        else:
            self.outpath = str(options.outpath)
            if not ( os.path.exists(self.outpath) ):
                os.mkdir( self.outpath )
                
        #check bxh if provided
        self.origbxh = None
        self.thisnii = None
        if options.funcfile is not None and self.needfunc:
            self.origbxh = str(options.funcfile)

        #t1
        self.t1bxh = None
        self.t1nii = None
        if options.anatfile is not None:
            fileExt = os.path.splitext(options.anatfile)[-1]
            if fileExt == '.bxh':
                self.t1bxh = str(options.anatfile)
            elif fileExt == '.gz' or fileExt == '.nii':
                self.t1nii = str(options.anatfile)
                runproc(str("fslwrapbxh " + self.t1nii))
                if os.path.isfile(self.t1nii.split('.')[0] + '.nii.gz'):
                    self.t1bxh = self.t1nii.split('.')[0] + '.nii.gz'
                elif os.path.isfile(self.t1nii.split('.')[0] + '.nii'):
                    self.t1bxh = self.t1nii.split('.')[0] + '.nii'

        if options.prefix is not None:
            self.prefix = str(options.prefix)
        else:
                    (funchead, functail) = os.path.split(str(options.funcfile))
                    if functail.endswith('.nii.gz'):
                        self.prefix = functail[0:-7]
                        self.thisnii = options.funcfile
                    elif functail.endswith('.nii'):
                        self.prefix = functail[0:-4]
                        self.thisnii = options.funcfile
                    else:
                        comps = functail.split('.')
                        if len(comps) == 1:
                            self.prefix = functail
                        else:
                            self.prefix = '.'.join(comps[0:-2])

        #grab TR if it needs to be forced
        self.tr_ms = options.tr_ms

        #set a basedir where this file is located
        self.basedir = re.sub('\/bin','',os.path.dirname(os.path.realpath(__file__)))
        #grab compcor setting
        self.compcor_components=options.compcor_components
        
        #grab detrend setting
        self.detrend = options.detrend
            
        #regressors
        self.regressors = options.regressors
        if 'compcor' in self.regressors:
            if 'csf' not in self.regressors or 'wm' not in self.regressors:
                print("If using CompCor, at least one tissue type must be selected as a regressor.")
                raise SystemExit()
        
        #grab masks
        self.gmmask = options.gmmask
        self.wmmask = options.wmmask
        self.csfmask = options.csfmask
        
        #grab flirt cost for bold to mni registrations
        self.flirtcost = options.flirtcost
        #grab regmethod
        self.regmethod = options.regmethod
        #reference image for normalization
        self.refac = str(options.refac)
        logging.info('Using ' + options.refac + ' for AC point/centroid calculation')
        
        if options.template is not None:
            if options.sstemplate is None:
                print("Please provide a skull stripped image of your template. This can be done with 3dSkullStrip or BET.")
                raise SystemExit()
            else:
                self.template=str(options.template)
        else:
            self.template = os.path.join(self.basedir,'data','MNI152_T1_2mm.nii.gz')
            
        if options.sstemplate is not None:
            if self.regmethod == 'fsl' and self.t1nii is not None and options.template is None:
                 print("If using not using default template, a skull-on image is required for FNIRT.")
                 raise SystemExit()
            if self.regmethod == 'fsl' and self.t1nii is not None and options.fnirtbrainmask is None and options.space == 'Template':
                 print("If using not using default template, a brain mask is required for FNIRT when obtaining derivatives in Template space.")
                 raise SystemExit()     
            self.sstemplate = str(options.sstemplate)
        else:
            self.sstemplate = os.path.join(self.basedir,'data','MNI152_T1_2mm_brain.nii.gz')
             
        if options.refbrainmask is not None:
            self.refbrainmask = str(options.refbrainmask)
        else:
            self.refbrainmask = os.path.join(self.basedir,'data','MNI152_T1_2mm_brain_mask.nii.gz')
        
        if options.fnirtbrainmask is not None:
            self.fnirtbrainmask = str(options.fnirtbrainmask)
        else:
            self.fnirtbrainmask = os.path.join(self.basedir,'data','MNI152_T1_2mm_brain_mask_dil.nii.gz')
            
        if options.fnirtconfig is not None:
            self.fnirtconfig = str(options.fnirtconfig)
        else: 
            self.fnirtconfig = os.path.join(self.basedir,'data','T1_2_MNI152_2mm.cnf')
            
        if ( '0' in self.steps ) and (self.origbxh is None) and ( self.thisnii is not None ):
            if self.tr_ms is not None:                
                logging.info('requesting step0, but no bxh provided.  Creating one from ' + self.thisnii )
                runproc(str("fslwrapbxh " + self.thisnii))
                tmpfname = re.split('(\.nii$|\.nii\.gz$)',self.thisnii)[0] + ".bxh"

                if os.path.isfile( tmpfname ):
                    self.origbxh = tmpfname
                else:
                    logging.info("BXH creation failed")
                    raise SystemExit()
            else:
                logging.info("Please provide --tr option when starting from nifti, we don't trust TR derrived from existing nifti files.")
                raise SystemExit()
        
        #grab FWHM 
        self.fwhm = options.fwhm
        
        #check original space calculations
        self.space=options.space

        #grab correlation label, or assign the AAL brain
        if options.corrlabel is not None:
            self.corrlabel = str(options.corrlabel)
            if options.corrtext is not None:
                self.corrtext  = str(options.corrtext)
            else:
                logging.info("Please provide --corrtext option when the default template is not being used.")
                raise SystemExit()
        else:
            self.corrlabel = os.path.join(self.basedir,'data','aal_MNI_V4.nii')
            self.corrtext = os.path.join(self.basedir,'data','aal_MNI_V4.txt')
        
        #grab motion parameters if needed
        self.mcparams = options.mcparams
        
        #grab band-pass filter input
        self.lpfreq = options.lpfreq
        self.hpfreq = options.hpfreq

        #grab skull stripping options
        if options.fval is not None:
            self.fval = options.fval
        else:
            if options.skullstrip == 'afni' or options.skullstrip =='ants':
                self.fval = 0.5
            elif options.skullstrip == 'bet':
                self.fval = 0.4
        
        if options.anatfval is not None:
            self.anatfval = options.anatfval 
            self.shrinkfac= '-shrink_fac ' + str(self.anatfval)
        else:
            if options.skullstrip == 'afni':
                self.anatfval = 0.6
                self.shrinkfac= '-shrink_fac ' + str(self.anatfval)
            elif options.skullstrip == 'bet':
                self.anatfval = 0.5     

        #grab remaining options
        self.segmenttransform = options.segmenttransform
        self.sliceorder = options.sliceorder
        self.throwaway = options.throwaway
        self.sst1=None
        self.unsst1=None
        self.oldt1nii = None
        self.t1maskbinarypath=None
        self.dofleft=None
        self.oldnii = self.thisnii
        self.scrubop = 'or'
        self.resamplet1 = options.resamplet1
        self.dvarsthreshold = None
        self.dvarsnumneighbors = 0
        self.fdthreshold = None
        self.fdnumneighbors = 0
        self.motionthreshold = None
        self.motionnumneighbors = 1
        self.fcdmthresh = 0.6
        if options.powerscrub:
            # these override any previously set options
            self.scrubop = 'and'
            self.dvarsthreshold = '0.5%'
            self.dvarsnumneighbors = 0
            self.fdthreshold = 0.5
            self.fdnumneighbors = 0
        if options.scrubop is not None:
            self.scrubop = options.scrubop
        if options.dvarsthreshold is not None:
            self.dvarsthreshold = options.dvarsthreshold
        if self.dvarsthreshold is not None:
            checkval = self.dvarsthreshold
            if checkval[-1] == '%':
                checkval = checkval[0:-1]
            try:
                _ = float(checkval)
            except:
                logging.error("--dvarsthreshold must be a floating-point number (optionally followed by '%')")
                raise SystemExit()
        if options.dvarsnumneighbors is not None:
            self.dvarsnumneighbors = options.dvarsnumneighbors
        if options.fdthreshold is not None:
            self.fdthreshold = options.fdthreshold
        if options.fdnumneighbors is not None:
            self.fdnumneighbors = options.fdnumneighbors
        if options.motionthreshold is not None:
            self.motionthreshold = options.motionthreshold
        if options.motionnumneighbors is not None:
            self.motionnumneighbors = options.motionnumneighbors
        self.scrubkeepminvols = options.scrubkeepminvols
        self.mcparams = options.motionpar
        if self.motionthreshold != None:
            if '2' not in self.steps:
                if options.motionpar == None:
                    logging.info("--motionpar option is required when using --motionthreshold if you are skipping the motion correction step (step 2).")
                    raise SystemExit()
                self.mcparams = options.motionpar
        if options.fcdmthresh is not None:
            self.fcdmthresh = float(options.fcdmthresh)        
        
        #array for files to delete later
        self.toclean = []
        self.slicefile = None
        #self.sliceorder = None
        self.xdim = None
        self.ydim = None
        self.zdim = None    
        self.tdim = None
        #self.thisnii = None
        self.prevprefix = None
                
        #make normalization outpath
        if '3' or '4' in self.steps:      
            self.regoutpath = os.path.join(self.outpath, 'NormalizationFiles',)
            if not (os.path.exists(self.regoutpath)):
                os.mkdir( self.regoutpath)
        self.segoutpath = os.path.join(self.outpath, 'SegmentationFiles')    
        if not os.path.isdir(self.segoutpath) and'5' in self.steps:
            os.makedirs(self.segoutpath)
       
        #place to put temp stuff
        if ( os.getenv('TMPDIR') ):
            self.tmpdir = os.getenv('TMPDIR')
        else:
            self.tmpdir = '/tmp'

        #if they are skipping 0, make sure there's NII data
        if '0' not in self.steps and self.needfunc:
            if self.thisnii is None:                
                newfile = os.path.join(self.outpath,self.prefix)
                runproc(str("bxh2analyze --overwrite --niigz -s " + self.origbxh + " " + newfile))
                if os.path.isfile(newfile + ".nii.gz"):
                    self.thisnii = newfile + ".nii.gz"

            if self.t1nii is None and self.t1bxh is not None:
                fileName = self.t1bxh.split('/')[-1].split('.')[0]
                newfile = os.path.join(self.outpath,fileName)
                runproc(str("bxh2analyze --overwrite --niigz -s " + self.t1bxh + " " + newfile))
                if os.path.isfile(newfile + ".nii.gz"):
                    self.t1nii = newfile + ".nii.gz"

                if os.path.isfile(newfile + ".bxh"):
                    self.t1bxh = newfile + ".bxh"
        #Check tr and BOLD dimension      
        if (self.thisnii is not None) and ( self.tr_ms is not None ):
            thishdr = nibabel.load(self.thisnii).get_header()
            thisshape = thishdr.get_data_shape()

            if len(thisshape) == 4:
                self.xdim = thisshape[0]
                self.ydim = thisshape[1]
                self.zdim = thisshape[2]
                self.tdim = thisshape[3]
            else:
                logging.info("Functional data has incorrect dimensions. Expected 4D, received : " + str(len(thisshape)) + " D")
                raise SystemExit()
        elif ( [ step for step in ['0', '1', '6', '7', '8'] if step in self.steps ] ):
            logging.info("Please provide --tr option when starting from nifti, we don't trust TR derrived from existing nifti files.")
            raise SystemExit()
            
        #try to determine sliceorder if step1
        if '1' in self.steps:
            if re.search('\d+\,+',str(self.sliceorder)):
                slicefile = os.path.join(self.outpath,'sliceorder.txt')
                f = open(slicefile, 'w')
                for i in self.sliceorder.split(','):
                    f.write(i)
                    f.write("\n")

                f.close()
                self.slicefile = slicefile
            elif options.sliceorder:
                #try to generate a slicefile if it wasn't in BXH
                if (re.search('(odd|even|up|down)',options.sliceorder)) and (self.slicefile is None):
                    self.sliceorder = options.sliceorder                    
                    if self.zdim is not None:
                        slicefile = os.path.join(self.outpath,'sliceorder.txt')
                        f = open(slicefile, 'w')
                        if self.sliceorder == 'up': #bottomup
                            thisrang = list(range(1,self.zdim + 1))
                            for i in thisrang:
                                f.write(str(i))
                                f.write("\n")
                            
                            f.close()
                            self.slicefile = slicefile
                        elif self.sliceorder == 'down': #topdown
                            thisrang = list(range(1,self.zdim + 1))
                            thisrang.reverse() #flip it
                            for i in thisrang:
                                f.write(str(i))
                                f.write("\n")
                            
                            f.close()
                            self.slicefile = slicefile
                        elif re.search('(odd|even)',self.sliceorder):  #interleaved
                            odds = list(range(1,self.zdim+1,2))
                            evens = list(range(2,self.zdim+1,2))
                            if self.sliceorder == 'odd': #odds first
                                for i in odds:
                                    f.write(str(i))
                                    f.write("\n")

                                for i in evens:
                                    f.write(str(i))
                                    f.write("\n")
                            elif self.sliceorder == 'even': #evens first
                                for i in evens:
                                    f.write(str(i))
                                    f.write("\n")

                                for i in odds:
                                    f.write(str(i))
                                    f.write("\n")

                            f.close()
                            self.slicefile = slicefile
                    else:
                        logging.info("z dimension could not be found.")
                        raise SystemExit()
                else:
                    logging.info("sliceorder is incorrectly defined. use odd/even/up/down.")
                    raise SystemExit()
            else:
                logging.info("slice order not found. please use --sliceorder option")
                raise SystemExit()
            
        #PRE FLIGHT CHECK    
        #make sure motion regression is possible     
        if 'motion' in self.regressors and '2' not in self.steps and options.mcparams is None and '6' in self.steps:
            logging.info("Motion regression requested, but motion regressors not found and not being calculated. Please provide motion regressors, run step 2, or skip motion regression.")
            raise SystemExit()
        
        #check space calculations
        if self.space == 'T1' or self.space == 'BOLD':
            if '4' not in self.steps:
                if options.corrlabel is None or options.corrtext is None:
                    logging.info("You requested derivatives in BOLD or T1 subject space, but are not running normalization and did not provide a ROI label files in the subject space. Please run step 4 or provide --corrlabel and --corrtext." )
                    raise SystemExit()
                if '5' in self.steps and options.sstemplate is None and self.t1nii is None:
                    logging.info("You requested derivatives in BOLD or T1 subject space, but are not running normalization and did not provide a template files in the subject space. Please run step 4 or provide --ssref or a T1 image in the analysis space." )
                    raise SystemExit()
            else:
                if self.t1nii is None and self.space == 'T1':
                    logging.info("You requested derivatives in T1 subject space, but did not provide a T1 file" )
                    raise SystemExit()
      
        #check masks
        if self.gmmask is None and '5' not in self.steps and '11' in self.steps:
            logging.info('Please provide a gray matter mask or run segmentation.')
            raise SystemExit()
        if ((self.wmmask or self.csfmask) is None) and ('5' not in self. steps) and (('wm' or 'csf') in self.regressors) and '6' in self.steps:
            logging.info('Please run segmentation or provide WM and CSF masks in the ANALYSIS SPACE for regression.')
            raise SystemExit()        
        if ((self.wmmask or self.csfmask or self.gmmask) is not None) and ('5' in self. steps):
            logging.info('The tissue masks provided will not be used, as segmentation is being performed.')    
        if '3' not in self.steps and '4' in self.steps and self.t1nii is not None and self.regmethod == 'fsl':
            logging.info('Invalid settings. Step 3 required.')
            raise SystemExit()
        
        #check segment transform is available
        if '5' in self.steps and '4' not in self.steps and self.segmenttransform is None:
            if self.space == 'BOLD' or (self.space == 'Template' and self. t1nii is not None):
                logging.info('Please provide a transformation file or run registration.')
                raise SystemExit()               
        if self.segmenttransform is not None and '4' in self.steps:
            logging.info('Registration will overrule the transformation that was provided for segmentation.') 
        
        #check segment transform file type
        if self.segmenttransform is not None and not (self.segmenttransform.endswith('.mat') or self.segmenttransform.endswith('.nii') or self.segmenttransform.endswith('.nii.gz')):
            logging.info('Wrong file type %s', self.segmenttransform)
            raise SystemExit()
        
        # If running step 10b by itself, check corrts now
        self.corrts = None
        if len(self.steps) == 1 and '10b' in self.steps:
            # running step 9b by itself.  See if --corrts is specified or
            # otherwise look for default parcellation output file
            if options.corrts is not None:
                if not os.path.isfile(options.corrts):
                    print("File does not exist: " + options.corrts)
                    raise SystemExit()
                self.corrts = options.corrts
            else:
                corrtsfile = os.path.join(self.outpath,'corrlabel_ts.txt')
                if not os.path.isfile(corrtsfile):
                    print("You are running step 10b by itself, but can't find default input file '%s'.  Please specify an alternate file with --corrts." % (corrtsfile,))
                    raise SystemExit()
        
    #function to get the labels from the text file
    def grab_labels(self):
        mylabs = open(self.corrtext,'r').readlines()
        labs = []

        for line in mylabs:
            if len(line.split('\t')) == 2:
                splitstuff = line.split('\t')
                splitstuff[0] = int(splitstuff[0])
                splitstuff[-1] = splitstuff[-1].strip()
                labs.append(splitstuff)

        return labs        
       
    #step0 is the initial LAS conversion and nifti creation    
    def step0(self):        
        logging.info('converting functional data')
        tempfile = os.path.join(self.tmpdir,''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(10)) + '.bxh')
        runproc(str("bxhreorient --orientation=LAS " + self.origbxh + " " + tempfile))

        if os.path.isfile(tempfile):
            newprefix = self.prefix + "_LAS"
            newfile = os.path.join(self.outpath,newprefix)
            runproc(str("bxh2analyze --overwrite --niigz -s " + tempfile + " " + newfile))
            if os.path.isfile(newfile + ".nii.gz"):
                self.thisnii = newfile + ".nii.gz"
                self.prevprefix = self.prefix
                self.prefix = newprefix                 
            else:
                logging.info('conversion failed')
                raise SystemExit()
        else:
            logging.info('orientation change failed')
            raise SystemExit()

        if self.t1bxh is not None or self.t1nii is not None:
            logging.info('converting anatomical data')
            newprefix = "t1_LAS"
            newfile = os.path.join(self.outpath,newprefix)
            runproc(str("bxhreorient --orientation=LAS " + self.t1bxh + " " + newfile + ".bxh"))

            if os.path.isfile(newfile + ".nii.gz"):
                self.t1nii = newfile + ".nii.gz"
            else:
                logging.info('anatomical conversion failed')
                raise SystemExit()

    #slice time correction
    def step1(self):
        
        if self.throwaway is not None:
            logging.info('Disregarding acquisitions')
            newprefix = self.prefix + "_ta"
            newfile = os.path.join(self.outpath,newprefix)
            runproc(str("bxhselect --overwrite --timeselect " + str(self.throwaway) + ": " + self.thisnii + " " + newfile))
            if self.tdim < self.throwaway*10:
                print('WARNING: You are disregardign over 10% of your timepoints. Please verify that is correct.')
            self.tdim = self.tdim - self.throwaway
            if os.path.isfile(newfile + ".nii.gz"):
                self.oldnii = self.thisnii
                self.thisnii = newfile + ".nii.gz"
                self.prevprefix = self.prefix
                self.prefix = newprefix
            else:
                logging.info('throwaway failed')
                raise SystemExit()
            
        logging.info('slice time correcting data')
        newprefix = self.prefix + '_st'
        newfile = os.path.join(self.outpath,newprefix)
        
        runproc(str("slicetimer -i " + self.thisnii + " -o " + newfile + " -r " +  str(self.tr_ms/1000) + " --ocustom=" + self.slicefile))
        
        if os.path.isfile(newfile + ".nii.gz"):
            if self.prevprefix is not None:
                self.toclean.append( self.thisnii )
            self.thisnii = newfile + ".nii.gz"
            self.prevprefix = self.prefix
            self.prefix = newprefix
            logging.info('slice time correction successful')
        else:
            logging.info('slice time correction failed')
            raise SystemExit()

    #run motion correction
    def step2(self):
        logging.info('motion correcting data')
        newprefix = self.prefix + '_mcf'
        newfile = os.path.join(self.outpath,newprefix)
        
        runproc(str("mcflirt -in " + self.thisnii + " -o " + newfile + " -plots"))

        if os.path.isfile(newfile + ".nii.gz") and os.path.isfile(newfile + ".par"):
            if self.prevprefix is not None:
                self.toclean.append( self.thisnii )
            self.oldnii = self.thisnii
            self.thisnii = newfile + ".nii.gz"
            self.prevprefix = self.prefix
            self.prefix = newprefix
            if self.mcparams is None:
                self.mcparams = newfile + ".par"
            logging.info('motion correction successful: ' + self.thisnii )
            #make plots
            runproc(str("fsl_tsplot -i " + self.mcparams +  " -t 'MCFLIRT estimated rotations (radians)' -u 1 --start=1 --finish=3 -a x,y,z -w 640 -h 144 -o " + newfile + "_rot.png"))
            runproc(str("fsl_tsplot -i " + self.mcparams +  " -t 'MCFLIRT estimated translations (mm)' -u 1 --start=4 --finish=6 -a x,y,z -w 640 -h 144 -o " + newfile + "_trans.png"))

        else:
            logging.info('motion correction failed')
            raise SystemExit()

    #skull strip 
    def step3(self):
        
        #skull strip the functional
        newprefix = self.prefix + "_brain"
        newfile = os.path.join(self.outpath, newprefix)

        if options.skullstrip == 'afni' or options.skullstrip == 'ants':
            logging.info('Skull stripping func using AFNI')
            boldstrip = afni.Automask(in_file=self.thisnii, out_file=os.path.join(self.outpath,'func_brain_mask.nii.gz'), brain_file=str(newfile + '.nii.gz'), clfrac=self.fval, terminal_output='none')
            boldstrip.run()
                            
            #pictures to check bold skull strip
            self.meanbold = mean_img(self.thisnii)
            self.meanfuncbrain = mean_img(str(newfile+'.nii.gz'))
            display = plotting.plot_img((self.meanbold), cmap=plt.cm.Greens, cut_coords=(0,0,0))
            display.add_overlay((self.meanfuncbrain), cmap=plt.cm.Reds, alpha=0.6)
            display.savefig(os.path.join(self.regoutpath, 'SS_BOLD.png'))
            
        elif options.skullstrip == 'bet':
            logging.info('Skull stripping func using BET')
            #first create mean_func
            runproc(str("fslmaths " + self.thisnii + " -Tmean " + os.path.join(self.outpath,'mean_func')))
            #now skull strip the mean
            runproc(str("bet " + os.path.join(self.outpath,'mean_func') + " " + os.path.join(self.outpath,'mean_func_brain') + " -f " + str(self.fval) + " -m"))
            #now mask full run by results
            runproc(str("fslmaths " + self.thisnii + " -mas " + os.path.join(self.outpath,'mean_func_brain_mask') + " " + newfile))
                            
            #pictures to check bold skull strip
            self.meanfuncbrain = os.path.join(self.outpath,'mean_func_brain.nii.gz')
            display = plotting.plot_img(os.path.join(self.outpath,'mean_func.nii.gz'), cmap=plt.cm.Greens, cut_coords=(0,0,0))
            display.add_overlay(self.meanfuncbrain, cmap=plt.cm.Reds, alpha=0.6)
            display.savefig(os.path.join(self.regoutpath, 'SS_BOLD.png'))

        if os.path.isfile( newfile + ".nii.gz" ):
            if self.prevprefix is not None:
                self.toclean.append( self.thisnii )
            self.toclean.append( os.path.join(self.outpath,'mean_func.nii.gz') )
            self.oldnii = self.thisnii
            self.thisnii = newfile + ".nii.gz"
            self.prevprefix = self.prefix
            self.prefix = newprefix
            logging.info('Skull stripping completed: ' + self.thisnii )
        else:
            logging.info('Skull stripping failed')
            raise SystemExit()

        #skull strip anat
        if self.t1nii is not None:
            newprefix = self.t1nii.split('/')[-1].split('.')[0] + "_brain"
            newfile = os.path.join(self.outpath, newprefix)
            self.maskprefix = self.t1nii.split('/')[-1].split('.')[0] + "_brain" + "_mask"
            self.maskfile = os.path.join(self.outpath, self.maskprefix)
            self.t1maskbinary = self.maskfile + "_binary"
            
            if options.skullstrip == 'afni':
                logging.info('Skull stripping anatomical using AFNI.')
                t1mask = afni.SkullStrip(in_file=self.t1nii, out_file=str(self.maskfile + '.nii.gz'), terminal_output='none', args = self.shrinkfac + " -mask_vol")
                t1mask.run()
                t1strip = afni.Calc(in_file_a=self.t1nii, in_file_b=str(self.maskfile + '.nii.gz'),expr='a*step(b)', out_file = str(newfile + '.nii.gz'), terminal_output='none')
                t1strip.run()
                t1maskbinary =afni.Calc(in_file_a=str(self.maskfile + '.nii.gz'), expr='ispositive(a-0.9999)', out_file = str(self.t1maskbinary + '.nii.gz'), terminal_output='none', args='-datum byte')
                t1maskbinary.run()
            elif options.skullstrip == 'bet':
                logging.info('Skull stripping anatomical using BET.')
                runproc(str("bet " + self.t1nii + " " + newfile + " -f " + str(self.anatfval) + " -m"))
                self.t1maskbinary=newfile + "_mask"
            elif options.skullstrip == 'ants':
                logging.info('Skull stripping anatomical using ANTs.')
                self.ants_sstemplate = os.path.join(self.basedir,'data','OASIS_template.nii.gz')
                self.ants_ssprob = os.path.join(self.basedir,'data','OASIS_template_prob.nii.gz')
                self.ants_ssregmask = os.path.join(self.basedir,'data','OASIS_template_regmask.nii.gz')
                runproc(str("antsBrainExtraction.sh -d 3 -a " + self.t1nii + " -e " + self.ants_sstemplate + " -m " + self.ants_ssprob + " -f " + self.ants_ssregmask + " -o " + newfile))            
                self.t1maskbinary = newfile + 'BrainExtractionMask'
                newfile = newfile + 'BrainExtractionBrain'
                
            if os.path.isfile( newfile + ".nii.gz" ):
                self.unsst1=self.t1nii
                self.t1nii = newfile + ".nii.gz"
                self.t1maskbinarypath = self.t1maskbinary + ".nii.gz"
                logging.info('Skull stripping completed: ' + self.t1nii )
                
                #pictures to check t1 skull strip
                display = plotting.plot_img(self.unsst1, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.t1nii, cmap=plt.cm.Reds, alpha=0.4)
                display.savefig(os.path.join(self.regoutpath, 'SS_T1.png'))
            else:
                logging.info('Skull stripping anatomical failed.')
                raise SystemExit()
                
    #normalize the data
    def step4(self):
        logging.info('Normalizing data.')
        newprefix = self.prefix + "_norm"
        newfile = os.path.join(self.outpath, newprefix)           
        out_file = newfile + '.nii.gz'
        
    #Resample T1
        if self.t1nii is not None:
            if self.resamplet1 == 'yes':
                logging.info('Resampling T1.')
                #resample skull stripped
                t1prefix = self.t1nii.split('/')[-1].split('.')[0] + "_resampled.nii.gz"
                self.sst1_resampled = os.path.join(self.outpath, t1prefix)         
                reference= ants.image_read(self.sstemplate)
                moving=ants.image_read(self.t1nii) 
                resampled=ants.resample_image(moving, reference.spacing, False, 0)
                ants.image_write(resampled, self.sst1_resampled)
                self.oldt1nii = self.t1nii
                self.t1nii = self.sst1_resampled
                
                if self.unsst1 is not None:
                    #resample non skull stripped
                    unsst1prefix = self.unsst1.split('/')[-1].split('.')[0] + "_resampled.nii.gz"
                    self.unsst1_resampled = os.path.join(self.outpath, unsst1prefix)
                    reference= ants.image_read(self.sstemplate)
                    moving=ants.image_read(self.unsst1) 
                    resampled=ants.resample_image(moving, reference.spacing, False, 0)
                    ants.image_write(resampled, self.unsst1_resampled)
                    self.unsst1 = self.unsst1_resampled
                
                if self.t1maskbinarypath is not None:
                    #resample binary mask
                    self.t1maskbinary_resampled=self.t1maskbinary + "_resampled"
                    self.t1maskbinarypath_resampled=self.t1maskbinary + "_resampled.nii.gz"
                    reference= ants.image_read(self.sstemplate)
                    moving=ants.image_read(self.t1maskbinarypath) 
                    resampled=ants.resample_image(moving, reference.spacing, False, 0)
                    ants.image_write(resampled, self.t1maskbinarypath_resampled)
                    self.t1maskbinary = self.t1maskbinary_resampled
                    self.t1maskbinarypath = self.t1maskbinarypath_resampled
        
	#ANTs
        if self.regmethod == 'ants':
            self.meanfuncbrain = os.path.join(self.outpath,'mean_func_brain.nii.gz')
            if os.path.isfile(self.meanfuncbrain) == False:
                    #first create mean_func
                    logging.info('Mean fuctional not found, creating mean funcional.')
                    runproc(str("fslmaths " + self.thisnii + " -Tmean " + os.path.join(self.outpath,'mean_func_brain')))
                    self.meanfuncbrain=os.path.join(self.outpath,'mean_func_brain.nii.gz')
                    
            if self.space == 'Template':             
                self.t1normalized=os.path.join(self.regoutpath,'t1normalized'+'.nii.gz')
                self.boldcoregistered=os.path.join(self.regoutpath,'boldcoregistered'+'.nii.gz')
                self.toclean.append( self.t1normalized ) 
                self.toclean.append( self.boldcoregistered ) 
                if self.t1nii is not None:
                    self.sst1=self.t1nii #save for later image creation
                    #func to T1 rigid registration
                    logging.info('ANTs func to t1')
                    moving = ants.image_read(self.meanfuncbrain) #mean func
                    fixed=ants.image_read(self.t1nii)
                    reference = ants.image_read(self.sstemplate)
                    fixed=ants.resample_image(fixed,reference.shape,True,0)
                    moving=ants.resample_image(moving,reference.shape,True,0)
                    tx_func2t1 = ants.registration(fixed=fixed, moving=moving, type_of_transform='BOLDAffine')
                
                    #apply the transform
                    logging.info('Coregistering func')
                    moving=ants.image_read(self.thisnii) #orig func
                    coregistered = ants.apply_transforms(fixed=fixed, moving=moving, imagetype=3, transformlist=tx_func2t1['fwdtransforms'])
                    ants.image_write(coregistered, self.boldcoregistered)
                
                    #SyN the t1 to standard
                    logging.info('ANTs t1 to standard')
                    fixed = ants.image_read(self.sstemplate)
                    moving = ants.image_read(self.t1nii)
                    moving=ants.resample_image(moving,reference.shape,True,0)
                    tx_t12standard = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN', outprefix=os.path.join(self.regoutpath, 'ants_')) 
                    t1normalizedimg = tx_t12standard['warpedmovout']
                    ants.image_write(t1normalizedimg, self.t1normalized)
    
                    #apply the transform
                    logging.info('Normalizing func')
                    fixed = ants.image_read(self.sstemplate)
                    moving=ants.image_read(self.boldcoregistered) #4d fmri in t1 space  
                    vector=reference.shape + (moving.shape[3],)
                    moving=ants.resample_image(moving,vector,True,0)
                    normalized = ants.apply_transforms(fixed=fixed, moving=moving, imagetype=3, transformlist=tx_t12standard['fwdtransforms'])
                    ants.image_write(normalized, out_file)
                    
                    #save transform for segmentation
                    self.segmenttransform=tx_t12standard['fwdtransforms']
                    
                    #Images to check normalization
                    #bold on t1
                    self.meanboldcoregistered =  mean_img(self.boldcoregistered)
                    display = plotting.plot_img(self.meanboldcoregistered, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sst1, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'BOLDtoT1.png'))
                    #t1 on template
                    display = plotting.plot_img(self.t1normalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sstemplate, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'T1toTEMPLATE.png'))
                    #bold on template
                    self.meanboldnormalized =  mean_img(out_file)
                    display = plotting.plot_img(self.meanboldnormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sstemplate, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'BOLDtoTEMPLATE.png'))
                    
                else:
                    #use the functional to get the matrix
                    #func to Template affine + nonlinear registration
                    logging.info('ANTs func to template')
                    moving = ants.image_read(self.meanfuncbrain) #mean func
                    fixed=ants.image_read(self.sstemplate)
                    moving=ants.resample_image(moving,fixed.shape,True,0)
                    tx_func2template = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyNBoldAff')
                    
                    #then apply, as for some reason it doesn't output 4D data above
                    logging.info('Normalizing func')
                    moving=ants.image_read(self.thisnii) #orig func
                    normalized = ants.apply_transforms(fixed=fixed, moving=moving, imagetype=3, transformlist=tx_func2template['fwdtransforms'])
                    ants.image_write(normalized, out_file)
                    
                    #Images to check normalization
                    self.meanboldnormalized =  mean_img(out_file)
                    #bold on template
                    display = plotting.plot_img(self.meanboldnormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sstemplate, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'BOLDtoTEMPLATE.png'))
                    
            elif self.space == 'T1':
                
                self.templatenormalized=os.path.join(self.regoutpath,'templatenormalized'+'.nii.gz')
                self.toclean.append( self.templatenormalized )               
                self.boldcoregistered=out_file
                self.subjcorrlabel=os.path.join(self.outpath,'labelsinT1space'+'.nii.gz')
                self.sst1=self.t1nii #save for later image creation
                
                #func to T1 rigid registration
                logging.info('ANTs func to t1')
                moving = ants.image_read(self.meanfuncbrain) #mean func
                fixed=ants.image_read(self.t1nii)
                moving=ants.resample_image(moving,fixed.shape,True,0)
                tx_func2t1 = ants.registration(fixed=fixed, moving=moving, type_of_transform='BOLDAffine')
                
                #apply the transform
                logging.info('Coregistering func')
                moving=ants.image_read(self.thisnii) #orig func
                coregistered = ants.apply_transforms(fixed=fixed, moving=moving, imagetype=3, transformlist=tx_func2t1['fwdtransforms'])
                ants.image_write(coregistered, self.boldcoregistered)
                
                #SyN the t1 to standard
                logging.info('ANTs template to T1')
                fixed = ants.image_read(self.t1nii)
                moving = ants.image_read(self.sstemplate)
                moving=ants.resample_image(moving,fixed.shape,True,0)
                tx_standard2t1 = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
                templatenormalizedimg = tx_standard2t1['warpedmovout']
                ants.image_write(templatenormalizedimg, self.templatenormalized)
                
                #apply the transform
                logging.info('Normalizing label file')
                fixed = ants.image_read(self.t1nii)
                moving=ants.image_read(self.corrlabel) #label file in template space 
                moving=ants.resample_image(moving, fixed.shape,True,0)
                normalized = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_standard2t1['fwdtransforms'])
                ants.image_write(normalized, self.subjcorrlabel)
                self.corrlabel=self.subjcorrlabel
                
                #make images to check normalization
                self.meanboldcoregistered =  mean_img(self.boldcoregistered)
                #bold on t1  
                display = plotting.plot_img(self.meanboldcoregistered, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.sst1, cmap=plt.cm.Reds, alpha=0.4)
                display.savefig(os.path.join(self.regoutpath, 'BOLDtoT1.png'))
                #template on t1
                display = plotting.plot_img(self.templatenormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.sst1, cmap=plt.cm.Reds, alpha=0.4)
                display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoT1.png'))
                #template on bold in t1
                display = plotting.plot_img(self.templatenormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.meanboldcoregistered, cmap=plt.cm.Reds, alpha=0.4)
                display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoT1onBOLD.png'))
                #labels on bold in t1
                display = plotting.plot_img(self.corrlabel, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.meanboldcoregistered, cmap=plt.cm.Reds, alpha=0.3)
                display.savefig(os.path.join(self.regoutpath, 'LABELStoT1onBOLD.png'))
                
            elif self.space == 'BOLD':
                self.t1coregistered=os.path.join(self.regoutpath,'t1coregistered'+'.nii.gz')
                self.templateont1=os.path.join(self.regoutpath,'templateont1'+'.nii.gz')
                self.templatenormalized=os.path.join(self.regoutpath,'templatenormalized'+'.nii.gz')
                self.subjcorrlabel=os.path.join(self.outpath,'labelsinBOLDspace'+'.nii.gz')
                self.toclean.append( self.t1coregistered )
                self.toclean.append( self.templateont1 )
                self.toclean.append( self.templatenormalized )
                
                #update bold despite no transfomation
                oldnii=ants.image_read(self.thisnii)
                ants.image_write(oldnii, out_file)  
                
                if self.t1nii is not None:
                    self.sst1=self.t1nii #save for later image creation
                   
                    #SyN the template to T1
                    logging.info('ANTs template to T1')
                    fixed = ants.image_read(self.t1nii)
                    moving = ants.image_read(self.sstemplate)
                    moving=ants.resample_image(moving,fixed.shape,True,0)
                    tx_standard2t1 = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
                    template_t1img = tx_standard2t1['warpedmovout']
                    ants.image_write(template_t1img,self.templateont1)
                    
                    #T1 to BOLD rigid registration
                    logging.info('ANTs T1 to func')
                    fixed = ants.image_read(self.meanfuncbrain) #mean func
                    moving = ants.image_read(self.t1nii)
                    fixed=ants.resample_image(fixed,moving.shape,True,0)
                    tx_t12func = ants.registration(fixed=fixed, moving=moving, type_of_transform='BOLDAffine', outprefix=os.path.join(self.regoutpath, 'ants_'))
                    coregistered = tx_t12func['warpedmovout']
                    ants.image_write(coregistered, self.t1coregistered)
                    
                    #Apply to template
                    logging.info('Normalizing Template from T1 to BOLD')
                    fixed = ants.image_read(self.meanfuncbrain) #mean func
                    moving=ants.image_read(self.templateont1)
                    moving=ants.resample_image(moving, fixed.shape, True, 0)
                    templatenormalizedimg = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_t12func['fwdtransforms'])
                    ants.image_write(templatenormalizedimg, self.templatenormalized)
        
                    #Apply the transforms to label file
                    logging.info('Normalizing label file')
                    fixed = ants.image_read(self.t1nii) #mean func
                    moving=ants.image_read(self.corrlabel) #label file in template space 
                    moving=ants.resample_image(moving, fixed.shape,True,0)
                    normalized = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_standard2t1['fwdtransforms'])
                    fixed = ants.image_read(self.meanfuncbrain) #mean func
                    moving=ants.resample_image(normalized, fixed.shape,True,0)
                    normalized = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_t12func['fwdtransforms'])
                    ants.image_write(normalized, self.subjcorrlabel)
                    self.corrlabel=self.subjcorrlabel
                    
                    #save transform for segmentation
                    self.segmenttransform=tx_t12func['fwdtransforms']
                    
                    #make images to check normalization
                    self.meanbold =  mean_img(out_file)
                    #t1 on bold
                    display = plotting.plot_img(self.t1coregistered, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'T1toBOLD.png'))
                    #template on t1
                    display = plotting.plot_img(self.templateont1, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sst1, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoT1.png'))
                    #template on bold
                    display = plotting.plot_img(self.templatenormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoBOLD.png'))
                    #labels on bold
                    display = plotting.plot_img(self.corrlabel, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'LABELStoBOLD.png'))                   
                    
                else:
                    #use the functional to get the matrix
                    #template to bold affine + nonlinear registration
                    logging.info('ANTs Template to Func')
                    fixed = ants.image_read(self.meanfuncbrain) #mean func
                    moving=ants.image_read(self.sstemplate)
                    moving=ants.resample_image(moving,fixed.shape,True,0)
                    tx_template2func = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyNBoldAff', outprefix = os.path.join(self.regoutpath, 'ants_'))
                    templatenormalizedimg = tx_template2func['warpedmovout']
                    ants.image_write(templatenormalizedimg, self.templatenormalized)
                    
                    #normalize label file
                    logging.info('Normalizing label file')
                    moving=ants.image_read(self.corrlabel) #label file in template space 
                    moving=ants.resample_image(moving, fixed.shape,True,0)
                    normalized = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx_template2func['fwdtransforms'])
                    ants.image_write(normalized, self.subjcorrlabel)
                    self.corrlabel=self.subjcorrlabel

                    #save transform for segmentation
                    self.segmenttransform=tx_template2func['fwdtransforms']

                    #make images to check normalization
                    self.meanbold =  mean_img(out_file)
                    #template on bold
                    display = plotting.plot_img(self.templatenormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoBOLD.png'))
                    #labels on bold
                    display = plotting.plot_img(self.corrlabel, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'LABELStoBOLD.png'))

        #FSL
        elif self.regmethod == 'fsl':
            if self.space == 'Template':
                if self.t1nii is not None:
                    self.t1normalized=os.path.join(self.regoutpath,'t1normalized')
                    self.t1normalizedbrain=os.path.join(self.regoutpath,'t1normalized_brain'+'.nii.gz')
                    self.boldcoregistered=os.path.join(self.regoutpath,'boldcoregistered')
                    self.toclean.append(self.t1normalized)
                    self.toclean.append(self.t1normalizedbrain)
                    self.toclean.append(self.boldcoregistered)
                    self.sst1=self.t1nii
                        
                    #use t1 to generate flirt paramters
                    #first flirt the func to the t1
                    logging.info('flirt func to t1')
                    runproc(str("flirt -ref " + self.sst1 + " -in " + self.thisnii + " -out " + self.boldcoregistered + " -omat " + self.boldcoregistered + '.mat' + " -cost " + self.flirtcost + " -dof 6 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear"))
                    
                    #flirt the t1 to standard
                    logging.info('flirt t1 to standard')
                    runproc(str("flirt -ref " + self.sstemplate + " -in " + self.sst1 + " -out " + os.path.join(self.regoutpath,'t12standard_aff') + " -omat " + os.path.join(self.regoutpath,'t12standard_aff.mat') + " -cost corratio -dof 12 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear"))

                    #fnirt the t1 to standard
                    logging.info('fnirt t1 to standard')
                    runproc(str("fnirt --ref=" + self.template + " --refmask=" + self.fnirtbrainmask + " --in=" + self.unsst1 + " --aff=" + os.path.join(self.regoutpath,'t12standard_aff.mat') + " --config=" + self.fnirtconfig + " --iout=" + self.t1normalized + " --cout=" + os.path.join(self.regoutpath,'t12standard_fnirt_warpcoef')))
    
                    #apply the transform
                    logging.info('creating normalized func %s' % (newprefix))
                    runproc(str("applywarp --ref=" + self.sstemplate + " --in=" + self.thisnii + " --out=" + newfile + " --warp=" + os.path.join(self.regoutpath,'t12standard_fnirt_warpcoef.nii.gz') + " --premat=" + self.boldcoregistered + '.mat'))
                    
                    #apply the transform
                    logging.info('creating normalized t1 brain %s' % (self.t1normalizedbrain))
                    runproc(str("applywarp --ref=" + self.sstemplate + " --in=" + self.sst1 + " --out=" + self.t1normalizedbrain + " --warp=" + os.path.join(self.regoutpath,'t12standard_fnirt_warpcoef.nii.gz')))
                    
                    #save transform for segmentation
                    self.segmenttransform = os.path.join(self.regoutpath,'t12standard_fnirt_warpcoef.nii.gz')
                    
                    #check                   
                    if os.path.isfile(self.t1normalized + '.nii.gz'):      
                        self.t1normalized=self.t1normalized+'.nii.gz'                                 
                        logging.info('T1 normalization successful.')   
                    else:
                        logging.info('T1 normalization failed.')
                        raise SystemExit()
                    
                    #pictures for normalization
                    #bold on t1
                    self.boldcoregistered = self.boldcoregistered + '.nii.gz'
                    self.meanboldcoregistered =  mean_img(self.boldcoregistered)
                    display = plotting.plot_img(self.sst1, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanboldcoregistered, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'BOLDtoT1.png'))
                    #t1 on template
                    display = plotting.plot_img(self.t1normalizedbrain, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sstemplate, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'T1toTEMPLATE.png'))
                    #bold on template
                    self.meanboldnormalized =  mean_img(out_file)
                    display = plotting.plot_img(self.meanboldnormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sstemplate, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'BOLDtoTEMPLATE.png'))

                    
                else:
                    #use the functional to get the matrix
                    runproc(str("flirt -in " +  self.thisnii + " -ref " + self.sstemplate + " -out " + newfile + " -omat " + (newfile + '.mat') + " -bins 256 -cost " + self.flirtcost + " -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear"))
    
                    if os.path.isfile( newfile + '.mat' ):
                        #then apply output matrix to the same data with the same output name. for some reason flirt doesn't output 4D data above
                        logging.info('applying transformation matrix to 4D data')
                        runproc(str("flirt -in " + self.thisnii + " -ref " + self.sstemplate + " -applyxfm -init " + (newfile + '.mat') + " -out " + newfile))
                    else:
                        logging.info('Creation of initial flirt matrix failed.')
                        raise SystemExit()
                        
                    #pictures for normalization
                    self.meanboldnormalized =  mean_img(out_file)
                    display = plotting.plot_img(self.meanboldnormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sstemplate, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'BOLDtoTEMPLATE.png'))

            elif self.space == 'T1':
                self.templatenormalized=os.path.join(self.regoutpath,'templatenormalized')
                self.templatenormalizedbrain=os.path.join(self.regoutpath,'templatenormalized_brain'+'.nii.gz')
                self.subjcorrlabel=os.path.join(self.outpath,'labelsinT1space'+'.nii.gz')
                self.toclean.append(self.templatenormalized)
                self.toclean.append(self.templatenormalizedbrain)
                self.sst1=self.t1nii
                    
                #use t1 to generate flirt paramters
                #first flirt the func to the t1
                logging.info('flirt func to t1')
                runproc(str("flirt -in " + self.thisnii + " -ref " + self.sst1 + " -out " + newfile + " -omat " + os.path.join(self.regoutpath,'boldcoregistered.mat') + " -cost " + self.flirtcost + " -dof 6 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear"))
                
                #apply to make it 4d
                if os.path.isfile( os.path.join(self.regoutpath,'boldcoregistered.mat') ):
                    #then apply output matrix to the same data with the same output name. for some reason flirt doesn't output 4D data above
                    logging.info('applying transformation matrix to 4D data')
                    runproc(str("flirt -in " + self.thisnii + " -ref " + self.sst1 + " -applyxfm -init " + (os.path.join(self.regoutpath,'boldcoregistered.mat')) + " -out " + newfile ))
                else:
                    logging.info('Creation of initial flirt matrix failed.')
                    raise SystemExit()
                
                #flirt the standard to t1
                logging.info('flirt standard to t1')
                runproc(str("flirt -ref " + self.sst1 + " -in " + self.sstemplate + " -out " + os.path.join(self.regoutpath,'standard2t1_aff') + " -omat " + os.path.join(self.regoutpath,'standard2t1_aff.mat') + " -cost corratio -dof 12 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear"))

                #fnirt the standard to t1
                logging.info('fnirt standard to t1')
                runproc(str("fnirt --ref=" + self.unsst1 + " --refmask=" + self.t1maskbinarypath + " --in=" + self.template + " --aff=" + os.path.join(self.regoutpath,'standard2t1_aff.mat') + " --config=" + self.fnirtconfig + " --iout=" + self.templatenormalized + " --cout=" + os.path.join(self.regoutpath,'standard2t1_fnirt_warpcoef')))
                
                #apply the transform
                logging.info('creating normalized labels %s' % (self.subjcorrlabel))
                runproc(str("applywarp --ref=" + self.sst1 + " --in=" + self.corrlabel + " --out=" + self.subjcorrlabel + " --warp=" + os.path.join(self.regoutpath,'standard2t1_fnirt_warpcoef.nii.gz')))
                self.corrlabel=self.subjcorrlabel                       
                
                #apply the transform
                logging.info('creating normalized template brain %s' % (self.templatenormalizedbrain))
                runproc(str("applywarp --ref=" + self.sst1 + " --in=" + self.sstemplate + " --out=" + self.templatenormalizedbrain + " --warp=" + os.path.join(self.regoutpath,'standard2t1_fnirt_warpcoef.nii.gz')))

                #check
                if os.path.isfile(self.templatenormalized + '.nii.gz'):
                    self.templatenormalized=self.templatenormalized+'.nii.gz'
                    logging.info('Template normalization successful.')  
                else:
                    logging.info('Template normalization failed.')
                    raise SystemExit()
                
                #make images to check normalization
                self.boldcoregistered = newfile + '.nii.gz'
                #bold on t1  
                self.meanboldcoregistered =  mean_img(self.boldcoregistered)
                display = plotting.plot_img(self.meanboldcoregistered, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.sst1, cmap=plt.cm.Reds, alpha=0.4)
                display.savefig(os.path.join(self.regoutpath, 'BOLDtoT1.png'))
                #template on t1
                display = plotting.plot_img(self.templatenormalizedbrain, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.sst1, cmap=plt.cm.Reds, alpha=0.4)
                display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoT1.png'))
                #template on bold in t1
                display = plotting.plot_img(self.templatenormalizedbrain, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.meanboldcoregistered, cmap=plt.cm.Reds, alpha=0.4)
                display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoT1onBOLD.png'))
                #labels on bold in t1
                display = plotting.plot_img(self.corrlabel, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                display.add_overlay(self.meanboldcoregistered, cmap=plt.cm.Reds, alpha=0.3)
                display.savefig(os.path.join(self.regoutpath, 'LABELStoT1onBOLD.png'))           
                
            elif self.space == 'BOLD':
                self.subjcorrlabel=os.path.join(self.outpath,'labelsinBOLDspace.nii.gz')
                self.templatenormalized=os.path.join(self.regoutpath,'templatenormalized')
                self.toclean.append(self.templatenormalized)
                #update bold despite no transformation
                oldnii=nibabel.load(self.thisnii)
                nibabel.save(oldnii, out_file)
                
                if self.t1nii is not None:
                    self.t1coregistered=os.path.join(self.regoutpath,'t1coregistered')
                    self.templateont1=os.path.join(self.regoutpath,'templateont1')
                    self.templateont1brain=os.path.join(self.regoutpath,'templateont1_brain.nii.gz')
                    self.templatenormalizedbrain=os.path.join(self.regoutpath,'templatenormalized_brain.nii.gz')
                    self.toclean.append(self.templateont1brain)
                    self.toclean.append(self.templatenormalizedbrain)
                    self.sst1=self.t1nii
                
                    #use t1 to generate flirt paramters
                    #first flirt the t1 to func
                    if self.flirtcost == 'bbr':
                        print("WARNING: BBR cost is selected, so tranform will be BOLD->T1 and the inverse will be applied to the Template.")
                        runproc(str("flirt -in " +  self.thisnii + " -ref " + self.sst1 + " -out " +  self.t1coregistered + " -omat " + ( self.t1coregistered + '.mat') + " -bins 256 -cost " + self.flirtcost + " -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear"))
                        runproc(str("convert_xfm -omat " + self.templatenormalized + '.mat -inverse ' +  self.t1coregistered + '.mat'))
                        runproc(str("flirt - ref " + self.thisnii + " -in " + self.sst1 + " -applyxfm -init " +   self.t1coregistered + '.mat -out ' +  self.t1coregistered))
                    else:
                        logging.info('flirt T1 to BOLD')
                        runproc(str("flirt -ref " + self.thisnii + " -in " + self.sst1 + " -out " + self.t1coregistered + " -omat " + self.t1coregistered + '.mat' + " -cost  " + self.flirtcost + " -dof 6 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear"))
               
                    #flirt the standard to t1
                    logging.info('flirt standard to t1')
                    runproc(str("flirt -ref " + self.sst1 + " -in " + self.sstemplate + " -out " + os.path.join(self.regoutpath,'standard2t1_aff') + " -omat " + os.path.join(self.regoutpath,'standard2t1_aff.mat') + " -cost corratio -dof 12 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear"))
        
                    #fnirt the standard to t1
                    logging.info('fnirt standard to t1')
                    runproc(str("fnirt --ref=" + self.unsst1 + " --refmask=" + self.t1maskbinarypath + " --in=" + self.template + " --aff=" + os.path.join(self.regoutpath,'standard2t1_aff.mat') + " --config=" + self.fnirtconfig + " --iout=" + self.templateont1 + " --cout=" + os.path.join(self.regoutpath,'standard2t1_fnirt_warpcoef')))
                
                    #apply the transform
                    logging.info('creating normalized template %s' % (self.templatenormalized))
                    runproc(str("applywarp --ref=" + self.thisnii + " --in=" + self.sstemplate + " --out=" + self.templatenormalized + " --warp=" + os.path.join(self.regoutpath,'standard2t1_fnirt_warpcoef.nii.gz') + " --postmat=" + self.t1coregistered + '.mat'))
                    
                    #apply the transform
                    logging.info('creating normalized labels %s' % (self.subjcorrlabel))
                    runproc(str("applywarp --ref=" + self.thisnii + " --in=" + self.corrlabel + " --out=" + self.subjcorrlabel + " --warp=" + os.path.join(self.regoutpath,'standard2t1_fnirt_warpcoef.nii.gz') + " --postmat=" + self.t1coregistered + '.mat'))
                    self.corrlabel=self.subjcorrlabel
                    
                    #apply the transform
                    logging.info('creating template on t1  %s' % (self.templateont1brain))
                    runproc(str("applywarp --ref=" + self.thisnii + " --in=" + self.sst1 + " --out=" + self.templateont1brain + " --warp=" + os.path.join(self.regoutpath,'standard2t1_fnirt_warpcoef.nii.gz')))
                    
                    #save transform for segmentation
                    self.segmenttransform=self.t1coregistered + '.mat'
                    
                    #check
                    if os.path.isfile(self.templatenormalized + '.nii.gz') and os.path.isfile(self.templateont1 + '.nii.gz'):   
                        self.templatenormalized=self.templatenormalized + '.nii.gz'
                        self.templateont1=self.templateont1 + '.nii.gz'
                        self.toclean.append(self.templateont1)                 
                        logging.info('T1 normalization successful.')
                    else:
                        logging.info('t1 normalization failed.')
                        raise SystemExit()   
                    
                    #make images to check normalization
                    self.meanbold =  mean_img(self.thisnii)
                    self.t1coregistered= self.t1coregistered + '.nii.gz'
                    self.toclean.append(self.t1coregistered)
                    #t1 on bold
                    display = plotting.plot_img(self.t1coregistered, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'T1toBOLD.png'))
                    #template on t1
                    display = plotting.plot_img(self.templateont1brain, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.sst1, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoT1.png'))
                    #template on bold
                    display = plotting.plot_img(self.templatenormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.4)
                    display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoBOLD.png'))
                    #labels on bold
                    display = plotting.plot_img(self.corrlabel, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanbold, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'LABELStoBOLD.png'))
                
                else:
                    #use the functional to get the matrix
                    if self.flirtcost == 'bbr':
                        print("WARNING: BBR cost is selected, so tranform will be BOLD->Template and the inverse will be applied to the Template.")
                        runproc(str("flirt -in " +  self.thisnii + " -ref " + self.sstemplate + " -out " + self.templatenormalized + " -omat " + (self.templatenormalized + '.mat') + " -bins 256 -cost " + self.flirtcost + " -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear"))
                        runproc(str("convert_xfm -omat " + self.templatenormalized + '.mat -inverse ' + self.templatenormalized + '.mat'))
                        runproc(str("flirt - ref " + self.thisnii + " -in " + self.sstemplate + " -applyxfm -init " +  self.templatenormalized + '.mat -out ' + self.templatenormalized))
                    else:
                        runproc(str("flirt -in " +  self.sstemplate + " -ref " + self.thisnii + " -out " + self.templatenormalized + " -omat " + (self.templatenormalized + '.mat') + " -bins 256 -cost " + self.flirtcost + " -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear"))
                    
                    if os.path.isfile( self.templatenormalized + '.mat' ):
                        #then apply to labels
                        logging.info('applying transformation matrix label file')
                        runproc(str("flirt -in " + self.corrlabel+ " -ref " + self.thisnii + " -applyxfm -init " + (self.templatenormalized + '.mat') + " -out " + self.subjcorrlabel))
                        self.corrlabel=self.subjcorrlabel
                        
                    else:
                        logging.info('Creation of initial flirt matrix failed.')
                        raise SystemExit()
                        
                    #save transform for segmentation
                    self.segmenttransform=self.templatenormalized + '.mat'
                     
                    #pictures for normalization
                    self.templatenormalized = self.templatenormalized +'.nii.gz'
                    self.meanboldnormalized =  mean_img(self.thisnii)
                    display = plotting.plot_img(self.templatenormalized, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanboldnormalized, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'TEMPLATEtoBOLD.png'))
                    
                    #pictures for normalization
                    self.meanboldnormalized =  mean_img(self.thisnii)
                    display = plotting.plot_img(self.corrlabel, cmap=plt.cm.Greens, cut_coords=(0,0,0))
                    display.add_overlay(self.meanboldnormalized, cmap=plt.cm.Reds, alpha=0.3)
                    display.savefig(os.path.join(self.regoutpath, 'LABELStoBOLD.png'))
        
        
        #Check
        if os.path.isfile( newfile + '.nii.gz' ):  
            self.oldnii = self.thisnii
            if self.prevprefix is not None:
                self.toclean.append( self.thisnii ) 
            self.thisnii = newfile + '.nii.gz'
            logging.info('initial normalization successful: ' + self.thisnii )

            self.prevprefix = self.prefix
            self.prefix = newprefix

            normshape = nibabel.nifti1.load(self.thisnii).shape
            if len(normshape) is 4:
                self.xdim = normshape[0]
                self.ydim = normshape[1]
                self.zdim = normshape[2]
                self.tdim = normshape[3]
            else:
                logging.info('normalized data has wrong shape, expecting 4D, received: ' + str(len(normshape)) + 'D')
                raise SystemExit()
        else:
            logging.info('normalization failed.')
            raise SystemExit()
        
    #tissue ssegmentation   
    def step5(self):
        logging.info('Segmenting tissue.')
  
        self.csfmask = os.path.join(self.segoutpath, 'mask_pve_0.nii.gz')
        self.gmmask = os.path.join(self.segoutpath, 'mask_pve_1.nii.gz')
        self.wmmask = os.path.join(self.segoutpath, 'mask_pve_2.nii.gz')
        self.masks = [self.csfmask, self.gmmask, self.wmmask]
        
        if self.oldt1nii is None and self.t1nii is not None:
            self.oldt1nii = self.t1nii
            
        if self.oldt1nii is not None:
            if self.resamplet1 == 'yes':
                if self.sst1_resampled is None:
                    self.sst1_resampled = os.path.join(self.outpath, 't1_resampled.nii.gz')                  
                reference= ants.image_read(self.sstemplate)
                moving=ants.image_read(self.oldt1nii) 
                resampled=ants.resample_image(moving, reference.spacing, False, 0)
                ants.image_write(resampled, self.sst1_resampled)
                logging.info('Running segmentation on resampled T1 image.')
                runproc(str("fast -t 1 -n 3 -o " + os.path.join(self.segoutpath,'mask') + " " + self.sst1_resampled))
            elif self.resamplet1 == 'no':
                logging.info('Running segmentation on original T1 image.')
                runproc(str("fast -t 1 -n 3 -o " + os.path.join(self.segoutpath,'mask') + " " + self.oldt1nii))       
            
            for mask in self.masks:
                if self.space == 'BOLD': 

                    logging.info('Converting segmentation to BOLD space.') 
                    if self.regmethod == 'ants':
                        self.meanfuncbrain = os.path.join(self.outpath,'mean_func_brain.nii.gz')
                        if os.path.isfile(self.meanfuncbrain) == False:
                            #first create mean_func
                            logging.info('Mean fuctional not found, creating mean funcional.')
                            runproc(str("fslmaths " + self.oldnii + " -Tmean " + os.path.join(self.outpath,'mean_func_brain')))
                            self.meanfuncbrain=os.path.join(self.outpath,'mean_func_brain.nii.gz')
                        fixed = ants.image_read(self.meanfuncbrain) #mean func
                        moving=ants.image_read(mask) #label file in template space 
                        moving=ants.resample_image(moving, fixed.shape,True,0)
                        normalized = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=self.segmenttransform)
                        ants.image_write(normalized, mask)
                    elif self.regmethod == 'fsl':
                        runproc(str('flirt -ref ' + self.oldnii + ' -in ' + mask + ' -applyxfm -init ' + self.segmenttransform + ' -out ' + mask ))
                if self.space == 'T1':
                    logging.info('Scaling segmentation.') 
                    fixed = ants.image_read(self.t1nii) #mean func
                    moving=ants.image_read(mask)
                    moving=ants.resample_image(moving, fixed.shape, True, 0)
                    ants.image_write(moving, mask)
                if self.space == 'Template':
                    logging.info('Converting segmentation to Template space.')
                    if self.regmethod == 'ants':
                        fixed = ants.image_read(self.sstemplate)
                        moving = ants.image_read(mask)
                        moving = ants.resample_image(moving,fixed.shape,True,0)
                        normalized = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=self.segmenttransform)
                        ants.image_write(normalized, mask)
                    elif self.regmethod == 'fsl':
                        runproc(str("applywarp --ref=" + self.sstemplate + " --in=" + mask + " --out=" + mask + " --warp=" + self.segmenttransform))
        else:
            logging.info('Running segmentation on the template image.')
            if options.sstemplate is not None:
                runproc(str("fast -t 1 -n 3 -o " + os.path.join(self.segoutpath,'mask') + " " + self.sstemplate)) #change this?
            else:
                self.csfmask = os.path.join(self.basedir,'data','MNI152_T1_2mm_brain_pve_0.nii.gz')
                self.gmmask = os.path.join(self.basedir,'data','MNI152_T1_2mm_brain_pve_1.nii.gz')
                self.wmmask = os.path.join(self.basedir,'data','MNI152_T1_2mm_brain_pve_2.nii.gz')
                self.masks = [self.csfmask, self.gmmask, self.wmmask]
            i=0
            for mask in self.masks:    
                if self.space == 'BOLD': 
                    logging.info('Converting segmentation to BOLD space.')
                    if self.regmethod == 'ants':
                        if os.path.isfile(self.meanfuncbrain) == False:
                            #first create mean_func
                            logging.info('Mean fuctional not found, creating mean funcional.')
                            runproc(str("fslmaths " + self.oldnii + " -Tmean " + os.path.join(self.outpath,'mean_func_brain')))
                            self.meanfuncbrain=os.path.join(self.outpath,'mean_func_brain.nii.gz')
                        fixed = ants.image_read(self.meanfuncbrain)
                        moving=ants.image_read(mask)
                        moving=ants.resample_image(moving, fixed.shape,True,0)
                        normalized = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=self.segmenttransform)      
                    
                        ants.image_write(normalized, os.path.join(self.segoutpath, 'mask_pve_'+str(i)+'.nii.gz'))

                    elif self.regmethod == 'fsl':                 
                        runproc(str('flirt -ref ' + self.oldnii + ' -in ' + mask + ' -applyxfm -init ' + self.segmenttransform + ' -out ' + os.path.join(self.segoutpath, 'mask_pve_' + str(i)+ '.nii.gz')))
                i=i+1
            if self.space == 'BOLD':
                self.csfmask = os.path.join(self.segoutpath, 'mask_pve_0.nii.gz')
                self.gmmask = os.path.join(self.segoutpath, 'mask_pve_1.nii.gz')
                self.wmmask = os.path.join(self.segoutpath, 'mask_pve_2.nii.gz')               
        
    #regress out nuissance variables
    def step6(self):
        logging.info('Nuissance regression.')   
        newprefix = self.prefix + '_regress'
        newfile = os.path.join(self.outpath,(newprefix + ".nii.gz"))          
        regressorsstr = " ".join(str(regressor) for regressor in self.regressors)
        print("Will regress " + regressorsstr)
        
        #Get regressors
        #Get motion regressor
        if 'motion' in self.regressors:
            #import parameters from options if specified. Overrules those obtained in step two
            if options.mcparams is not None:
                self.mcparams = options.mcparams  
                
        #Get WM regressor   
        if 'wm' in self.regressors:
            #mean time series for wm
            wmout = os.path.join(self.outpath,"wm_ts.txt")
            runproc(str("fslmeants -i " + self.thisnii + " -m " + self.wmmask + " -o " + wmout))
            if not os.path.isfile(wmout):
                    logging.info('Could not extract WM timeseries, quitting: ' + wmout)
                    raise SystemExit()
          
        #Get CSF regressor
        if 'csf' in self.regressors:       
            #mean time series for csf
            csfout = os.path.join(self.outpath,"csf_ts.txt")
            runproc(str("fslmeants -i " + self.thisnii + " -m " + self.csfmask + " -o " + csfout))
            if not os.path.isfile(csfout):
                    logging.info('Could not extract CSF timeseries, quitting: ' + csfout)
                    raise SystemExit()    
        
        #Get CompCor regressor     
        if 'compcor' in self.regressors:
            logging.info('Running CompCor.')
            compcor_masks = []
            compcor_out = os.path.join(self.outpath,"compcor_ts.txt")
            if 'wm' in self.regressors:
                compcor_masks.append(self.wmmask)
            if 'csf' in self.regressors:
                compcor_masks.append(self.csfmask)
            
            compcor = confounds.CompCor()
            compcor.inputs.realigned_file=self.thisnii
            compcor.inputs.components_file=compcor_out
            compcor.inputs.mask_files=compcor_masks
            compcor.inputs.merge_method='none'
            compcor.inputs.num_components=self.compcor_components
            compcor.inputs.regress_poly_degree=self.detrend
            compcor.inputs.repetition_time=self.tr_ms/1000
            compcor.run()
        
        #One step regression
        logging.info('Combining and regressing nuissance variables')
        self.regressparams = newfile + ".par"
        self.regressparamsmat = newfile +".mat"
        
        #load regressors and combine them
        if 'motion' in self.regressors:
            motion_ts=np.loadtxt(self.mcparams)
        if 'wm' in self.regressors:
            wm_ts = np.loadtxt(wmout)
            wm_ts = wm_ts[:,None]
        if 'csf' in self.regressors:
            csf_ts = np.loadtxt(csfout)
            csf_ts=csf_ts[:,None]
        if 'compcor' in self.regressors:
            compcor_ts = np.loadtxt(compcor_out, skiprows=1)
            
        if compcor in self.regressors:            
            regressors_ts = np.concatenate([regressor for regressor in [compcor_ts, motion_ts] if regressor.size > 0 ], axis=1)
        else:
            regressors_ts = np.concatenate([regressor for regressor in [wm_ts, csf_ts, motion_ts] if regressor.size > 0 ], axis=1)
        np.savetxt(self.regressparams, regressors_ts)
        
        #convert regressor params to .mat file
        runproc(str("Text2Vest " + self.regressparams + " " + self.regressparamsmat))

        #regress out data
        newprefix = self.prefix + 'r'
        newfile = os.path.join(self.outpath, (newprefix + ".nii.gz"))
        runproc(str("fsl_glm -i " + self.thisnii + " -d " + self.regressparamsmat + " --out_res=" + newfile))
        
        #Check degrees of freedom left
        if self.dofleft == None:
            self.dofleft=int(self.tdim)
        self.dofused = regressors_ts.shape[1]
        self.dofleft = self.dofleft - self.dofused

   
        if os.path.isfile(newfile):
            if self.prevprefix is not None:
                self.toclean.append(self.thisnii)
            self.prevprefix = self.prefix
            self.prefix = newprefix
            self.thisnii = newfile
            logging.info('Regression successful: ' + self.thisnii )
            logging.info('Used %d degrees of freedom. %d dof lef out of %d.' % (self.dofused, self.dofleft, self.tdim))
        else:
            logging.info('Regression failed')
            raise SystemExit()
        
    #detrending
    def step7(self):
        logging.info('Detrending data')
        newprefix = self.prefix + "_detr"
        newfile = os.path.join(self.outpath,(newprefix + ".nii.gz"))
        self.polort = '-polort ' + str(self.detrend)       
        detrender = afni.Detrend(in_file=self.thisnii, out_file=newfile, terminal_output='none', args=self.polort)
        detrender.run()

        if self.dofleft == None:
            self.dofleft=int(self.tdim)
        self.dofused = self.detrend + 1
        self.dofleft = self.dofleft - self.dofused

        if os.path.isfile(newfile):
            if self.prevprefix is not None:
                self.toclean.append(self.thisnii)

            self.prevprefix = self.prefix
            self.prefix = newprefix
            self.thisnii = newfile
            logging.info('Detrending successful: ' + self.thisnii )
            logging.info('Used %d degrees of freedom. %d dof lef out of %d.' % (self.dofused, self.dofleft, self.tdim))
		
        else:
            logging.info('Detrending failed')
            raise SystemExit()
            
    #bandpass filter
    def step8(self):
        logging.info('bandpass filtering data')
        newprefix = self.prefix + "_filt"
        newfile = os.path.join(self.outpath,(newprefix + ".nii.gz"))
        lpfreq = self.lpfreq        
        hpfreq = self.hpfreq

        bandpass = afni.Bandpass(in_file=self.thisnii, highpass=hpfreq, lowpass=lpfreq, despike=False, no_detrend=True, notrans=True, tr=self.tr_ms/1000, out_file=newfile, terminal_output='none', args=str("-mask " + self.thisnii))
        bandpass.run()
        
        if self.dofleft == None:
            self.dofleft= self.tdim
        self.dofused = (self.tdim * self.tr_ms/1000) * (self.lpfreq - self.hpfreq)
        self.dofleft = self.dofleft - self.dofused


        if os.path.isfile(newfile):
            if self.prevprefix is not None:
                self.toclean.append(self.thisnii)
            self.prevprefix = self.prefix
            self.prefix = newprefix
            self.thisnii = newfile
            logging.info('bandpass filtering successful: ' + self.thisnii )
            logging.info('Used %d degrees of freedom. %d dof lef out of %d.' % (self.dofused, self.dofleft, self.tdim))
		
        else:
            logging.info('bandpass filtering failed')
            raise SystemExit()
           
    #FWMH Smoothing
    def step9(self):
        logging.info('smoothing data')
        newprefix = self.prefix + "_smooth"
        newfile = os.path.join(self.outpath,(newprefix + ".nii.gz"))

        kernel_width = self.fwhm
        kernel_sigma = kernel_width/2.3548 
       
        #apply smoothing
        runproc(str("fslmaths " +  self.thisnii + " -kernel gauss " + str(kernel_sigma) + " -fmean " + newfile))
     
        if os.path.isfile(newfile):
            if self.prevprefix is not None:
                self.toclean.append( self.thisnii )
            self.prevprefix = self.prefix
            self.prefix = newprefix
            self.thisnii = newfile
            logging.info('smoothing completed: ' + self.thisnii )
        else:
            logging.info('smoothing failed')
            raise SystemExit() 
     
    #do the parcellation and correlation
    def step10(self):
        self.step10a()
        self.step10b()
        
    #do the parcellation
    def step10a(self):
        logging.info('starting parcellation')
        corrtxt = os.path.join(self.outpath,'corrlabel_ts.txt')

        runproc(str("fslmeants -i " + self.thisnii + " --label=" + self.corrlabel + " -o " + corrtxt ))
        if not os.path.isfile(corrtxt):
            logging.info('coudld not create mean timeseries matrix file')
            raise SystemExit()
           
       
    #do the correlation
    def step10b(self):
        logging.info('starting correlation')
        rmat = os.path.join(self.outpath,'r_matrix.nii.gz')
        rtxt = os.path.join(self.outpath,'r_matrix.csv')
        zmat = os.path.join(self.outpath,'zr_matrix.nii.gz')
        ztxt = os.path.join(self.outpath,'zr_matrix.csv')
        pmat = os.path.join(self.outpath,'p_matrix.nii.gz')
        ptxt = os.path.join(self.outpath,'p_matrix.csv')
        corrtxt = os.path.join(self.outpath,'corrlabel_ts.txt')
        maskname = os.path.join(self.outpath,'mask_matrix.nii.gz')
        graphml = os.path.join(self.outpath,'subject.graphml')

        if self.corrts != None:
            corrtxt = self.corrts

        if os.path.isfile(corrtxt):        
            timeseries = np.loadtxt(corrtxt,unpack=True)
            if not self.needfunc:
                self.tdim = timeseries.shape[1]
            if self.motionthreshold is not None or self.dvarsthreshold is not None or self.fdthreshold is not None:
                timeseries = self.scrub_motion_volumes(timeseries)

            #partial 
            t_timeseries = np.transpose(timeseries)     
            measure_partialcorr=ConnectivityMeasure(kind='partial correlation')   
            matrix_partialcorr = measure_partialcorr.fit_transform([t_timeseries])           
            matrix_partialcorrs = np.squeeze(matrix_partialcorr)

            
            #corr and z fisher corr
            myres = np.corrcoef(timeseries)
            myres = np.nan_to_num(myres)
            
            #convert corcoef
            with np.errstate(divide='ignore'):
                zrmaps = 0.5*np.log(np.divide((1+myres),(1-myres)))
            #find the inf vals on diagonal
            infs = (zrmaps == np.inf).nonzero()

            #replace the infs with 0            
            for idx in range(len(infs[0])):
                zrmaps[infs[0][idx]][infs[1][idx]] = 0
                
            #save     
            nibabel.save(nibabel.Nifti1Image(myres,None) ,rmat)
            nibabel.save(nibabel.Nifti1Image(zrmaps,None) ,zmat)   
            nibabel.save(nibabel.Nifti1Image(matrix_partialcorrs,None) ,pmat)
            np.savetxt(rtxt,myres,fmt='%f',delimiter=',')
            np.savetxt(ztxt,zrmaps,fmt='%f',delimiter=',')       
            np.savetxt(ptxt,matrix_partialcorrs,fmt='%f',delimiter=',')
            
            #create a mask for higher level, include everything below diagonal
            mask = np.zeros_like(myres)
            maskx,masky = mask.shape

            for idx in range(maskx):
                for idy in range(masky):
                    if idx > idy:
                        mask[idx][idy] = 1

            nibabel.save(nibabel.Nifti1Image(mask,None) ,maskname)

            labels = self.grab_labels()
            aalcenter = np.array(self.refac.split(','),dtype=int)
            
            labnii = nibabel.load(self.corrlabel)
            niidata = labnii.get_data()
            niihdr = labnii.get_header()
            zooms = np.array(niihdr.get_zooms())
            
            G=nx.Graph(atlas=str(self.corrlabel))
            for lab in labels:
                try:
                    #grab indices equal to label value
                    x,y,z = (niidata == lab[0]).nonzero()
                    centroid = np.array([int(x.mean()),int(y.mean()),int(z.mean())])
                    c_cent_str = str((centroid - aalcenter)*(zooms.astype('int')))[1:-1].strip()
                    lab.append( (centroid - aalcenter)*zooms.astype('int') )
                    lab.append( 0 )
                    G.add_node(lab[0],label=str(lab[1]),centroid=c_cent_str,intensityvalue=lab[0] )
                    timecourse = timeseries[int(lab[0] - 1)]
                    G.node[lab[0]]['timecourse'] = str(timecourse.tolist()).replace(',','').strip('\[\]')
                except:
                    pass

            sigx,sigy = (zrmaps != 0 ).nonzero()            
            for idx in range(len(sigx)):
                if sigx[idx] < sigy[idx]:
                    zrval = str(zrmaps[sigx[idx]][sigy[idx]])
                    rval = str(myres[sigx[idx]][sigy[idx]])
                    xidx = sigx[idx] + 1
                    yidx = sigy[idx] + 1
                    G.add_edge(xidx,yidx)
                    G.edges[xidx, yidx]['zrvalue'] = zrval
                    G.edges[xidx, yidx]['rvalue'] = rval

            B = nx.Graph.to_undirected(G)
            nx.write_graphml(B,graphml,encoding='utf-8', prettyprint=True)


            #check for the resulting files
            for fname in [rmat, zmat, maskname, ztxt, rtxt, graphml]:
                if os.path.isfile( fname ):
                    logging.info('correlation matrix finished : ' + fname)
                else:
                    logging.info('correlation failed')
                    raise SystemExit()                
        else:            
            logging.info('could not find mean timeseries matrix file "%s"', corrtxt)
            raise SystemExit()


    #fcdm
    def step11(self):
        import fcdm
        logging.info('starting functional connectivity density mapping')
           
        #load nifti data
        data = nibabel.nifti1.load(self.thisnii)
        #data1 = data.get_data()
        mask = nibabel.nifti1.load(self.gmmask)

        if mask.shape != data.shape[:-1]:
            logging.info('data and mask are different shapes!')
            raise SystemExit()
               
        logging.info("running %s, masked by %s, at pearsonr value of %f" % (self.thisnii, self.gmmask, self.fcdmthresh))
        outfile = fcdm.fcdm(self.thisnii, self.gmmask, self.fcdmthresh)

        if os.path.isfile(outfile):
            logging.info("fcdm results %s" % outfile)


    #make the cleanup step
    def cleanup(self):
        for fname in self.toclean:
            if os.path.isfile(fname):
                logging.info('deleting :' + fname )
                os.remove(fname)
        if os.path.isdir(self.segoutpath):
            shutil.rmtree(self.segoutpath)

    # given a sequence of unit quaternions, each of the form (qs,
    # [qv1, qv2, qv3]), compute their quaternion multiplication.
    # Returns tuple (qrs, [qrv1, qrv2, qrv3]) containing scalar and
    # vector of result.
    def combine_quaternions(self, qseq):
        if len(qseq) == 0:
            logging.error("combine_quaternions got an empty list")
            raise SystemExit()
        if len(qseq) == 1:
            return qseq[0]
        (qs1, qv1) = qseq[0]
        (qs2, qv2) = qseq[1]
        #print " combine_quaternions got: q1=[%.10g %.10g %.10g %.10g] q2=[%.10g %.10g %.10g %.10g]" % (qs1, qv1[0], qv1[1], qv1[2], qs2, qv2[0], qv2[1], qv2[2])
        qsr = (qs1 * qs2) - np.dot(qv1, qv2)
        #print "  qs1*qs2=%.10g qv1.qv2=%.10g" % ((qs1*qs2), np.dot(qv1,qv2))
        qvr = (qs1 * qv2) + (qs2 * qv1) + np.cross(qv1, qv2)
        #print "  qs1*qv2=%s qs2.qv1=%s qv1xqv2=%s" % (str((qs1*qv2)), str((qs2*qv1)), str(np.cross(qv1,qv2)))
        # normalize the new quaternion
        mag = math.sqrt(qsr*qsr + np.sum(np.power(qvr,2)))
        newseq = [ (qsr/mag, qvr/mag) ]
        #print "  returned %s" % str([newseq[0][0], newseq[0][1][0], newseq[0][1][1], newseq[0][1][2]])
        newseq.extend(qseq[2:])
        return self.combine_quaternions(newseq)

    # scrub volumes that exceed the motion threshold
    def scrub_motion_volumes(self, timeseries):
        motionmarkedvolstxt = os.path.join(self.outpath,'motiondisp_markedvols.txt')
        motiontxt = os.path.join(self.outpath,'motiondisp.txt')
        dvarsmarkedvolstxt = os.path.join(self.outpath,'dvars_markedvols.txt')
        fdmarkedvolstxt = os.path.join(self.outpath,'fd_markedvols.txt')
        dvarstxt = os.path.join(self.outpath,'dvars.txt')
        dvarspercenttxt = os.path.join(self.outpath,'dvarspercent.txt')
        fdtxt = os.path.join(self.outpath,'fd.txt')
        dvarsthreshtxt = os.path.join(self.outpath,'dvars_thresh.txt')
        excludedvolstxt = os.path.join(self.outpath,'total_excludedvols.txt')
        newprefix = "scrubbed_" + self.prefix
        newfile = os.path.join(self.outpath,(newprefix + ".nii.gz"))

        #load mcflirt params
        params = np.loadtxt(self.mcparams,unpack=True)

        # this stores how many metrics chose to exclude
        numexcls = [ 0 ] * params.shape[1]

        datanifti = nibabel.nifti1.load(self.thisnii)

        if self.dvarsthreshold != None:
            logging.info('calculating DVARS for: %s', self.thisnii)
            # masked array
            data = datanifti.get_data().astype(np.float64)
            maskdata = nibabel.nifti1.load(self.refbrainmask).get_data().astype(np.float64)
            #maskdata = nd.binary_erosion(maskdata, iterations=5)
            data = numpy.ma.array(data,
                                  mask=numpy.tile((maskdata == 0)[:,:,:,np.newaxis], (1, 1, 1, data.shape[3])))
            work = data.reshape([data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]])
            boldmean = np.ma.mean(work)
            work = np.ma.diff(work, axis=1)
            logging.debug("work.min=%s work.max=%s work.mean=%s", np.min(work), np.max(work), np.mean(work))
            #work = work / boldmean
            #logging.info("work.min=%s work.max=%s work.mean=%s", np.min(work), np.max(work), np.mean(work))
            work = work ** 2
            logging.debug("work.min=%s work.max=%s work.mean=%s", np.min(work), np.max(work), np.mean(work))
            work = np.ma.mean(work, axis=0)
            logging.debug("work.min=%s work.max=%s work.mean=%s", np.min(work), np.max(work), np.mean(work))
            dvars = np.hstack(([0], np.ma.sqrt(work)))
            logging.debug(' DVARS: %s', dvars)
            logging.debug("dvars.min=%s dvars.max=%s", np.min(dvars), np.max(dvars))
            boldscaling = 1.0
            if self.dvarsthreshold[-1] == '%':
                boldscaling = numpy.mean(boldmean) / 100.0
                _dvarsthreshold = float(self.dvarsthreshold[0:-1])
            else:
                _dvarsthreshold = float(self.dvarsthreshold)
            excludethese = np.nonzero(dvars > _dvarsthreshold * boldscaling)[0]
            excludethese = sorted(
                set([ excludethis
                      for ind in excludethese
                      for contributor in (ind,)
                      for excludethis in range(contributor - self.dvarsnumneighbors, contributor + self.dvarsnumneighbors + 1)
                      if excludethis < data.shape[3]
                     ]))
            logging.info(' marking these volumes due to DVARS > %g: %s', _dvarsthreshold * boldscaling, excludethese)            
            for excludethis in excludethese:
                numexcls[excludethis] += 1
            #with open(dvarsmarkedvolstxt, 'w') as f:
            f = open(dvarsmarkedvolstxt, 'w')
            f.write("# these are the volumes (indexed starting at 0) marked as exceeding the 'DVARS' threshold of %s\n" % self.dvarsthreshold)
            np.savetxt(f, np.transpose(numpy.array(excludethese)), fmt='%d', newline=' ')
            f.close()
            
            np.savetxt(dvarstxt, dvars, fmt='%g')
            np.savetxt(dvarspercenttxt, dvars / boldscaling, fmt='%g')
            np.savetxt(dvarsthreshtxt, [_dvarsthreshold * boldscaling], fmt='%f')

        if self.fdthreshold is not None:
            RT = np.array(params)
            # distance traveled by a voxel on the 50mm surface
            # rotated by an angle A is calculated using the law
            # of cosines:
            #   a^2 = b^2 + c^2 - 2bc*cos(A)
            # b == c, in this case, so:
            #   a = sqrt(2*(b^2)*(1 - cos(A)))
            RT_mm = np.array(RT)
            RT_mm[0:3,:] = np.sqrt(2*(50.*50.) * (1 - np.cos(RT_mm[0:3,:])))
            deltas = np.abs(np.diff(RT_mm, axis=1))
            sums = np.sum(deltas, axis=0)
            FD = np.hstack(([0], sums))
            excludethese = np.nonzero(FD > self.fdthreshold)[0]
            # Power et al. only exclude the later of the two volumes that
            # contribute to the DVARS spike, so contributor is (ind + 1,)
            excludethese = sorted(
                set([ excludethis
                      for ind in excludethese
                      for contributor in (ind + 1,)
                      for excludethis in range(contributor - self.fdnumneighbors, contributor + self.fdnumneighbors + 1)
                      if excludethis < data.shape[3]
                     ]))
            logging.info(' marking these volumes due to FD > %g mm: %s', self.fdthreshold, excludethese)            
            for excludethis in excludethese:
                numexcls[excludethis] += 1
            #with open(fdmarkedvolstxt, 'w') as f:
            f = open(fdmarkedvolstxt, 'w')
            f.write("# these are the volumes (indexed starting at 0) marked as exceeding the 'FD' threshold of %g mm\n" % self.fdthreshold)
            np.savetxt(f, np.transpose(numpy.array(excludethese)), fmt='%d', newline=' ')
            f.close()
            np.savetxt(fdtxt, FD, fmt='%g')

        if self.motionthreshold != None:
            maxdisplacements = [ 0 ] * params.shape[1]
            excludethese = set()
            # go through each pair of adjoining volumes
            for t in np.arange(params.shape[1] - 1):
                t1 = t
                t2 = t + 1
                # mcflirt provides rotation and translation parameters to
                # the reference volume.  We want to know rotation and
                # translation between two adjoining volumes, so we need to
                # combine them.  The rotation R is around the center of
                # gravity of the reference volume, unlike the affine
                # matrix where rotation is around the (0,0,0) corner, the
                # idea being that this maximally decouples the rotation
                # and translation parameters, which is perfect as we are
                # interested in finding the maximum displacement for each
                # component separately (see below).  So, to transform from
                # one volume with translation T1 and rotation R1 to
                # another volume (similarly with T2, R2), we need to
                # transform in this order: T1, R1, inv(R2), inv(T2).
                # Since both rotations are rigid rotations, we can
                # represent them using quaternions for simplicity in
                # combining them (especially since they themselves are
                # each composed of rotations around the three base axes).
                RT1 = params[:,t1]
                T1 = np.array(RT1[3:])
                R1 = np.array(RT1[0:3])
                RT2 = params[:,t2]
                T2 = np.array(RT2[3:])
                R2 = np.array(RT2[0:3])
                #print " R1 = %s" % (R1,)
                #print " T1 = %s" % (T1,)
                #print " R2 = %s" % (R2,)
                #print " T2 = %s" % (T2,)
                # Quaternions for each rotation, based on unit vectors x =
                # [1 0 0], y = [0 1 0], z = [0 0 1].  Scalar components
                # have _s, vectors have _v.
                qx1_s = math.cos(R1[0]/2.0)
                qx1_v = np.array([math.sin(R1[0]/2.0), 0, 0])
                qx2_s = math.cos(R2[0]/2.0)
                qx2_v = np.array([math.sin(R2[0]/2.0), 0, 0])
                qy1_s = math.cos(R1[1]/2.0)
                qy1_v = np.array([0, math.sin(R1[1]/2.0), 0])
                qy2_s = math.cos(R2[1]/2.0)
                qy2_v = np.array([0, math.sin(R2[1]/2.0), 0])
                qz1_s = math.cos(R1[2]/2.0)
                qz1_v = np.array([0, 0, math.sin(R1[2]/2.0)])
                qz2_s = math.cos(R2[2]/2.0)
                qz2_v = np.array([0, 0, math.sin(R2[2]/2.0)])
                #print " qx1=[%.10g,  %.10g, %.10g, %.10g]" % (qx1_s, qx1_v[0], qx1_v[1], qx1_v[2])
                #print " qy1=[%.10g,  %.10g, %.10g, %.10g]" % (qy1_s, qy1_v[0], qy1_v[1], qy1_v[2])
                #print " qz1=[%.10g,  %.10g, %.10g, %.10g]" % (qz1_s, qz1_v[0], qz1_v[1], qz1_v[2])
                #print " qx2=[%.10g,  %.10g, %.10g, %.10g]" % (qx2_s, qx2_v[0], qx2_v[1], qx2_v[2])
                #print " qy2=[%.10g,  %.10g, %.10g, %.10g]" % (qy2_s, qy2_v[0], qy2_v[1], qy2_v[2])
                #print " qz2=[%.10g,  %.10g, %.10g, %.10g]" % (qz2_s, qz2_v[0], qz2_v[1], qz2_v[2])
                # combined rotation R1 * inv(R2) (inverse of unit
                # quaternion is calculated by just negating the vector
                # component) where R1 and R2 are actually three combined
                # rotations themselves.  Note, though the rotations in
                # matrix form are applied Rx.Ry.Rz, in quaternion form
                # they have to be applied in the reverse order.
                (qR1_s, qR1_v) = self.combine_quaternions(((qz1_s, qz1_v), (qy1_s, qy1_v), (qx1_s, qx1_v)))
                (qR2_s, qR2_v) = self.combine_quaternions(((qz2_s, qz2_v), (qy2_s, qy2_v), (qx2_s, qx2_v)))
                (qcomb_s, qcomb_v) = self.combine_quaternions(((qR2_s, -1*qR2_v), (qR1_s, qR1_v)))
                #print " qcomb=[%g,  %g, %g, %g]" % (qcomb_s, qcomb_v[0], qcomb_v[1], qcomb_v[2])
                # OK, now our job is to find the maximum displacement
                # resulting from applying all the motion correction
                # parameters for any voxel 50mm from the center of
                # rotation.  This is as tricky as it sounds.  Basically,
                # for any of the voxels of interest (i.e., any voxels who
                # will undergo a rotation around a 50mm radius circle), we
                # have three displacement vectors that will be added to
                # determine the total displacement: (1) the first
                # translation to the reference volume (T1), (and to the
                # 50mm radius circle), (2) the result of the combined
                # rotation (qcombined), calculated above, and (3) the
                # translation to the neighbor volume (-1 * T2).  (1) and
                # (3) are the same for every voxel, so they can be added
                # together immediately.  (2) is the only one which changes
                # based on the voxel position.  However, we are only
                # interested in points on the 50mm radius circle, and
                # there is one point whose rotation displacement will have
                # the most impact on the total displacement: namely, the
                # point whose rotation displacement is parallel to the
                # projection P of the combined translation vector (T1 +
                # (-1 * T2)) onto the circle's plane.  We don't even need
                # to find the exact point, we just need to know that it
                # exists.  So we can just extend vector P to the magnitude
                # of the displacement caused by the rotation, and then add
                # this to the translation vector.  This gives us the
                # maximum displacement.
            
                u2scaled = np.array([0, 0, 0])
                if qcomb_s * qcomb_s != 1:
                    # there is rotation.
                    # find the angle and axis of rotation from the combined
                    # quaternion
                    angle = 2 * np.arccos(qcomb_s)
                    axis = np.array(qcomb_v) / (1 - (qcomb_s*qcomb_s))
                    # u1 is the projection of the (combined) translation onto
                    # the rotation axis, and u2 is the residual (i.e. the
                    # projection onto the circle's plane).
                    Tcomb = T1 + (-1 * T2)
                    #print " Tcomb=[%g, %g, %g]" % (Tcomb[0], Tcomb[1], Tcomb[2])
                    u1 = (np.inner(Tcomb, axis)/np.inner(axis, axis)) * axis
                    u2 = np.subtract(Tcomb, u1)
                    u2mag = math.sqrt(np.inner(u2, u2))
                    # find the magnitude of the rotation displacement on the
                    # 50mm radius circle, using the arbitrary point [x=50mm,
                    # y=0]
                    rotatedpoint = np.array([50 * math.cos(angle), 50 * math.sin(angle)])
                    magrot = math.sqrt(np.sum(np.power(np.subtract(np.array([50,0]), rotatedpoint), 2)))
                    # scale u2 to match magnitude of the rotation (magrot) and
                    # then add the translation vector to get the maximum
                    # displacement vector.
                    u2scaled = u2 * (magrot/u2mag)
                maxdisplacementvector = Tcomb + u2scaled
                maxdisplacement = math.sqrt(np.inner(maxdisplacementvector, maxdisplacementvector))
                maxdisplacements[t2] = maxdisplacement
                if maxdisplacement > self.motionthreshold:
                    for excludethis in range(t1 - self.motionnumneighbors, t2 + self.motionnumneighbors + 1):
                        if excludethis < 0:
                            continue
                        if excludethis >= self.tdim:
                            continue
                        excludethese.update([excludethis])
            excludethese = list(excludethese)
            logging.info(' marking these volumes due to motion > %g mm: %s', self.motionthreshold, excludethese)
            for excludethis in excludethese:
                numexcls[excludethis] += 1
            #with open(motionmarkedvolstxt, 'w') as f:
            f = open(motionmarkedvolstxt, 'w')
            f.write("# these are the volumes (indexed starting at 0) marked as exceeding the 'motion' threshold of %g\n" % self.motionthreshold)
            np.savetxt(f, numpy.array(excludethese)[:,np.newaxis], fmt='%d', newline=' ')
            f.close()
            np.savetxt(motiontxt, maxdisplacements, fmt='%g')

        if self.scrubop == 'and':
            # volumes will be selected if any of the
            # selected metrics allowed the volume
            selected = (np.array(numexcls) < (
                    (self.motionthreshold is not None) +
                    (self.dvarsthreshold is not None) +
                    (self.fdthreshold is not None)))
        elif self.scrubop == 'or':
            # volumes will be selected if all of the
            # selected metrics allowed the volume
            selected = (np.array(numexcls) == 0)

        timeseries = timeseries[:,np.array(selected)]
        excludedinds = np.array(np.nonzero(np.array(selected) == False))
        if len(excludedinds[0]) > 0:
            scrubmethodstr = (' ' + self.scrubop.upper() + ' ').join(
                ([], ["'DVARS'"])[self.dvarsthreshold is not None] +
                ([], ["'FD'"])[self.fdthreshold is not None] +
                ([], ["'motion'"])[self.motionthreshold is not None])
            logging.info('Scrubbed the following volumes because they or their neighbors exceeded the %s threshold (first volume is 0): %s' % (scrubmethodstr, str(excludedinds[0])))
            #with open(excludedvolstxt, 'w') as f:
            f = open(excludedvolstxt, 'w')
            f.write("# these are the volumes (indexed starting at 0) excluded because they or their neighbors exceeded the %s threshold\n" % scrubmethodstr)
            np.savetxt(f, np.transpose(excludedinds[0]), fmt='%d', newline=' ')
            f.close()

        if self.scrubkeepminvols != None and timeseries.shape[1] < self.scrubkeepminvols:
            logging.error('Too few volumes (%d) met the scrubbing threshold!  Exiting...' % (timeseries.shape[1],))
            raise SystemExit()

        # write out scrubbed image data (though we don't actually use it)
        scrubbeddata = datanifti.get_data()[:,:,:,np.array(selected)]
        newNii = nibabel.Nifti1Pair(scrubbeddata,None,datanifti.get_header())
        nibabel.save(newNii,newfile)

        if os.path.isfile(newfile):
            self.thisnii = newfile

        return timeseries


if __name__ == "__main__":
    pipeline = RestPipe()
#    pipeline.mainloop()
