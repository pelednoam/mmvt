function splitting_cortical()
    global SUBJECTS_DIR SUBJECT APARC FREESURFER_HOME
    params = load('params.mat');
    SUBJECT = params.subject
    SUBJECTS_DIR = params.subjects_dir
    APARC = params.aparc
    SCRIPTS = params.scripts_dir
    FREESURFER_HOME = params.freesurfer_home
    addpath(SCRIPTS)
    splitting_cortical_surface([SUBJECTS_DIR, '/', SUBJECT, '/', APARC, '.pial']);
    
    function splitting_cortical_surface(output_folder)
        hemis = {'lh', 'rh'};
        for hemi_ind=1:2
          hemi = hemis{hemi_ind};
          output_folder_hemi = [output_folder '.' hemi];
          if (~exist(output_folder_hemi, 'dir'))
              mkdir(output_folder_hemi);
          end
          chdir(output_folder_hemi);
          inner_splitting_cortical_surface(...
              [SUBJECTS_DIR '/' SUBJECT '/label/' hemi '.' APARC '.annot'], ...
              [SUBJECTS_DIR '/' SUBJECT '/label/' hemi '.' APARC '.annot.dpv'], ...
              [SUBJECTS_DIR '/' SUBJECT '/surf/' hemi '.pial.srf'], ... 
              [hemi '.pial.' APARC '.srf']);
        end
    end

    function inner_splitting_cortical_surface(annotation_file, dpv_output_file, ...
            subject_surface, file_prefix)
        annot2dpv(annotation_file, dpv_output_file, FREESURFER_HOME);
        splitsrf(subject_surface, dpv_output_file, file_prefix);
    end

end
