#! /bin/tcsh -f

#
# mris_compute_lgi
#
# Computes local measurements of gyrification at points over cortical surface.
# --help option will show usage
#
# This script implements the work of Marie Schaer et al., as described in:
# 
# "A Surface-based Approach to Quantify Local Cortical Gyrification",
# Schaer M. et al., IEEE Transactions on Medical Imaging, 2007, TMI-2007-0180
#
# Original Author: Marie Schaer
# CVS Revision Info:
#    $Author: nicks $
#    $Date: 2009/01/26 22:14:05 $
#    $Revision: 1.23 $
#
# Copyright (C) 2007-2008,
# The General Hospital Corporation (Boston, MA).
# All rights reserved.
#
# Distribution, usage and copying of this software is covered under the
# terms found in the License Agreement file named 'COPYING' found in the
# FreeSurfer source code root directory, and duplicated here:
# https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferOpenSourceLicense
#
# General inquiries: freesurfer@nmr.mgh.harvard.edu
# Bug reports: analysis-bugs@nmr.mgh.harvard.edu
#

set VERSION = '$Id: mris_compute_lgi,v 1.23 2009/01/26 22:14:05 nicks Exp $'
set PrintHelp = 0;
set RunIt = 1;
set cmdargs = ($argv);
set use_mris_extract = 1
set closespheresize = 15
set smoothiters = 30
set radius = 25
set stepsize = 100
#set echo=1
set start=`date`
set python_cmd = 'python -c "import sys; print(sys.executable)"'
echo $python_cmd

if($#argv == 0) then
  # zero args is not allowed
  goto usage_exit;
endif

#cd $SUBJECTS_DIR/surf

goto parse_args;
parse_args_return:

goto check_params;
check_params_return:

# begin...
#---------

# check for matlab
#set MATLAB = `getmatlab`;
# set MATLAB = 'octave'
# if($status) then
#   echo "ERROR: Matlab is required to run mris_compute_lgi!"
#   exit 1;
# endif

set rootdir = `dirname $0`
set abs_rootdir = `cd $rootdir && pwd`
echo "abs_rootdir:"
echo ${abs_rootdir}

set input_name = `python -c "import os.path as op; print('$input'.split(op.sep)[-1])"`
echo "input and input_name:"
echo ${input}
echo ${input_name}

# temporary work files go here...
set tmpdir = ($PWD/tmp-mris_compute_lgi/{$input_name})
echo "tmpdir:"
echo {$tmpdir}
set cmd=(rm -Rf $tmpdir)
echo "================="
echo "$cmd"
echo "================="
if ($RunIt) $cmd
set cmd=(mkdir -p $tmpdir)
echo "================="
echo "$cmd"
echo "================="
if ($RunIt) $cmd

#
# mris_fill
#
# create a filled-volume from the input surface file...
set cmd=(mris_fill -c -r 1 ${input} ${tmpdir}/${input_name}.filled.mgz)
echo "================="
echo "$cmd"
echo "================="
if ($RunIt) $cmd
if($status) then
  echo "ERROR: $cmd failed!"
  exit 1;
endif

if ! $?SCRIPTS_DIR then
    setenv SCRIPTS_DIR ${PWD}
endif

#
# make_outer_surface.m
#
# create the outer surface from the filled volume
set arg1 = ${tmpdir}/${input_name}.filled.mgz
set arg2 = ${closespheresize}
set arg3 =  ${tmpdir}/${input_name}-outer
echo "$python_cmd ${abs_rootdir}/mkoutersurf.py ${arg1} ${arg2} ${arg3}"
echo "================="
if ($RunIt) then
  pwd
  $python_cmd ${abs_rootdir}/mkoutersurf.py ${arg1} ${arg2} ${arg3}
endif
echo ""
if ( $RunIt && ! -e ${arg3} ) then
  echo "ERROR: make_outer_surface did not create output file '${arg3}'!"
  exit 1
endif

#
# mris_extract_main_component (optional, default)
#
if ($use_mris_extract) then
  set cmd=(mris_extract_main_component \
    ${tmpdir}/${input_name}-outer \
    ${tmpdir}/${input_name}-outer-main)
  echo "================="
  echo "$cmd"
  echo "================="
  if ($RunIt) $cmd
  if($status) then
    echo "ERROR: $cmd failed!"
    exit 1;
  endif
else
  set cmd=(cp ${tmpdir}/${input_name}-outer ${tmpdir}/${input_name}-outer-main)
  echo "================="
  echo "$cmd"
  echo "================="
  if ($RunIt) $cmd
  if($status) then
    echo "ERROR: $cmd failed!"
    exit 1;
  endif
endif

#
# mris_smooth
#
# smooth this jaggy, tessellated surface
set cmd=(mris_smooth -nw -n ${smoothiters} \
    ${tmpdir}/${input_name}-outer-main \
    ./${input_name}-outer-smoothed)
echo "================="
echo "$cmd"
echo "================="
if ($RunIt) $cmd
if($status) then
  echo "ERROR: $cmd failed!"
  exit 1;
endif


#--------
# end...
set cmd=(rm -Rf $tmpdir)
echo "================="
echo "$cmd"
echo "================="
if ($RunIt) $cmd
set end=`date`
echo "done."
echo "Start: $start"
echo "End:   $end"

mv ${input_name}-outer-smoothed `echo ${input_name} | cut -c1-3`dural

exit 0



#----------------------------------------------------------#
############--------------##################
parse_args:
set cmdline = ($argv);

while( $#argv != 0 )

  set flag = $argv[1]; shift;

  switch($flag)

   case "--close_sphere_size":
      set closespheresize = $argv[1]; shift;
      echo "Using sphere size of ${closespheresize}mm for morph closing op."
      breaksw

   case "--smooth_iters":
      set smoothiters = $argv[1]; shift;
      echo "Smoothing using ${smoothiters} iterations."
      breaksw

   case "--step_size":
      set stepsize = $argv[1]; shift;
      echo "Skipping every ${stepsize} vertices during lGI calcs."
      breaksw

    case "--help":
      set PrintHelp = 1;
      goto usage_exit;
      exit 0;
      breaksw

    case "--version":
      echo $VERSION
      exit 0;
      breaksw

    case "-i":
    case "--i":
    case "--input":
      if ( $#argv == 0) goto arg1err;
      set input = $argv[1]; shift;
      echo "input:"
      echo ${input}
      breaksw

   case "--dont_extract":
      set use_mris_extract = 0
      breaksw

   case "-p"
      set python_cmd = $argv[1]; shift;
      echo python_cmd:
      echo ${python_cmd}
      breaksw

   case "--debug":
   case "--echo":
      set echo = 1;
      set verbose = 1
      breaksw

   case "--dontrun":
      set RunIt = 0;
      breaksw

    default:
      breaksw

  endsw

end
goto parse_args_return;
############--------------##################



############--------------##################
arg1err:
  echo "ERROR: flag $flag requires one argument"
  exit 1
############--------------##################



############--------------##################
check_params:
  if(! $?FREESURFER_HOME ) then
    echo "ERROR: environment variable FREESURFER_HOME not set."
    exit 1;
  endif
  if(! -e $FREESURFER_HOME ) then
    echo "ERROR: FREESURFER_HOME $FREESURFER_HOME does not exist."
    exit 1;
  endif
  if(! $?input) then
    echo "ERROR: missing input filename!  See  mris_compute_lgi --help"
    exit 1;
  endif
  if(! -e ${input} ) then
    echo "ERROR: input file '${input}' does not exist."
    exit 1;
  endif
goto check_params_return;
############--------------##################



############--------------##################
usage_exit:
  echo ""
  echo "USAGE: mris_compute_lgi [options] --i <input surface>"
  echo ""
  echo "Produces a surface map file containing local gyrification measures."
  echo "Output file is named <input surface>_lgi, where <input surface> is the"
  echo "specified input surface (ex. lh.pial produces lh.pial_lgi)."
  echo ""
  echo "Required Arguments"
  echo "  --i       : input surface file, typically lh.pial or rh.pial"
  echo ""
  echo "Optional Arguments"
  echo "  --close_sphere_size <mm> : use sphere of size <mm> mm for morph"
  echo "                             closing operation (default: ${closespheresize})"
  echo "  --smooth_iters <iters>   : smooth outer-surface <iters> number of"
  echo "                             iterations (default: ${smoothiters})"
  echo "  --step_size <steps>      : skip every <steps> vertices when"
  echo "                             computing lGI (default: ${stepsize})"
  echo "  --help    : short descriptive help"
  echo "  --version : script version info"
  echo "  --echo    : enable command echo, for debug"
  echo "  --debug   : same as --echo"
  echo "  --dontrun : just show commands (dont run them)"
  echo ""

  if(! $PrintHelp) exit 1;

  echo Version: $VERSION

  cat $0 | awk 'BEGIN{prt=0}{if(prt) print $0; if($1 == "BEGINHELP") prt = 1 }'

exit 1;


#---- Everything below here is printed out as part of help -----#
BEGINHELP

Computes local measurements of gyrification at thousands of points over the 
entire cortical surface using the method described in:

"A Surface-based Approach to Quantify Local Cortical Gyrification",
Schaer M. et al., IEEE Transactions on Medical Imaging, 2007, TMI-2007-0180

Input is a pial surface mesh, and the output a scalar data file containing
the local gyrification index data at each vertices.

Example:

  mris_compute_lgi --i lh.pial

produces lh.pial_lgi

See also http://surfer.nmr.mgh.harvard.edu/fswiki/LGI
