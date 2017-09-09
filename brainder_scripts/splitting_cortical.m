function splitting_cortical()
    global SUBJECTS_DIR SUBJECT APARC FREESURFER_HOME
    params = load('params.mat');
    SUBJECT = params.subject
    SUBJECTS_DIR = params.subjects_dir
    MMVT_DIR = params.mmvt_dir
    APARC = params.aparc
    SCRIPTS = params.scripts_dir
    FREESURFER_HOME = params.freesurfer_home
    SURF_TYPE = params.surface_type
    addpath(SCRIPTS)
    splitting_cortical_surface([SUBJECTS_DIR, '/', SUBJECT, '/', APARC, '.' SURF_TYPE]);
    
    function splitting_cortical_surface(output_folder)
        hemis = {'lh', 'rh'};
        for hemi_ind=1:2
          hemi = hemis{hemi_ind};
          output_folder_hemi = [output_folder '.' hemi];
          if (~exist(output_folder_hemi, 'dir'))
              mkdir(output_folder_hemi);
          end
          chdir(output_folder_hemi);
          contours_faces = load([MMVT_DIR '/' SUBJECT '/contours_faces_' APARC '.mat']);
          inner_splitting_cortical_surface(...
              [SUBJECTS_DIR '/' SUBJECT '/label/' hemi '.' APARC '.annot'], ...
              [SUBJECTS_DIR '/' SUBJECT '/label/' hemi '.' APARC '.annot.dpv'], ...
              [MMVT_DIR '/' SUBJECT '/surf/' hemi '.' SURF_TYPE '.mat'], ... 
              [hemi '.' SURF_TYPE '.' APARC], ...
              contours_faces.(hemi));
              %[SUBJECTS_DIR '/' SUBJECT '/surf/' hemi '.' SURF_TYPE '.srf'], ... 
        end
    end

    function inner_splitting_cortical_surface(annotation_file, dpv_output_file, ...
            subject_surface, file_prefix, contours_faces)
        ret = annot2dpv(annotation_file, dpv_output_file, FREESURFER_HOME);
        if ret
            splitsrf(subject_surface, dpv_output_file, file_prefix, contours_faces);
        end
    end

end
