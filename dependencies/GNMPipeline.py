import os
import shutil
from glob import glob
from nilearn import plotting
import subprocess
import sys
import matplotlib_inline
from tqdm import tqdm
import tempfile
from pathlib import Path
from time import sleep

script_dir = os.path.dirname(os.path.abspath(__file__))

def print_tree(d, n=5, indent=0):
    """
    Recursively prints the folder structure.
    
    Parameters:
    d (dict): The folder structure dictionary.
    n (int): The maximum number of subjects to print. Default is 5.
    indent (int): The indentation level (number of spaces). Default is 0.
    """
    
    subset = {k: d[k] for k in list(d)[:n]}
    
    for key, value in subset.items():
        print('    ' * indent + str(key))
        if isinstance(value, list):
            for item in value:
                print('    ' * (indent + 1) + str(item))
        elif isinstance(value, dict):
            print_tree(value, indent + 1)
            
def create_input_dict(input_folder, subjects_to_skip=None, input_type='Folder'):
    
    """
    Creates a dictionary of subjects and their corresponding sessions from a given input folder.

    This function looks at the files in the input folder and creates a dictionary where the keys are the subject IDs and the values are lists of sessions for each subject.

    Parameters:
    input_folder (str): The path to the folder that contains the input files.
    subjects_to_skip (list): A list of subject IDs that you want to skip. Defaults to None.
    input_type (str): The type of input folder. Can be either 'BIDS' or 'Folder'. Defaults to 'Folder'.

    Returns:
    subject_sessions (dict): A dictionary where the keys are the subject IDs and the values are lists of sessions for each subject.

    Note:
    If input_type is 'BIDS', the function assumes that the input folder is organized according to the BIDS format.
    If input_type is 'Folder', the function assumes that all selected scans are in one folder and the file names start with the subject ID followed by an underscore.
    """
    
    if not Path(input_folder).is_dir():
        raise ValueError("Input folder is not a valid directory")
    
    subject_sessions = {}
    
    if input_type=='BIDS':
        subjects=[f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
        
        if subjects_to_skip is not None:
            subjects = [subject for subject in subjects if subject not in subjects_to_skip]
            
        print("There are", len(subjects), "unique subjects to be registered")
        
        for subject in sorted(subjects):
            # Get the path to the subject folder
            subject_path = os.path.join(input_folder, subject)

            # Get a list of session folders within the subject folder
            sessions = [f for f in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, f))]

            # Add the subject and sessions to the dictionary
            subject_sessions[subject] = sessions


        print(subject_sessions)
        
    
    elif input_type=='Folder':
        #All selected scans are in one folder
        #Assumptions: first part of file name is subject ID followed by an _
        subjects=sorted(set([os.path.basename(i).split('_')[0] for i in glob(f'{input_folder}/*.nii*')]))
        
        if subjects_to_skip is not None:
            subjects = [subject for subject in subjects if subject not in subjects_to_skip]
            
        print("There are", len(subjects), "unique subjects to be registered")
        
        
        for subject in sorted(subjects):
            for file in glob(f'{input_folder}/*{subject}*.nii*'):
                subject = os.path.basename(file).split('_')[0]
                session = os.path.basename(file).split('_')[1]

                if subject not in subject_sessions:
                    subject_sessions[subject] = []
                if session not in subject_sessions[subject]:  
                    subject_sessions[subject].append(session)


        print(subject_sessions)

    
    else:
        raise ValueError(f"Invalid input_type: '{input_type}'. Should be either 'BIDS' or 'Folder'.")
    
    
    return subject_sessions
 
def submit_slurm_job(job_name, command, partition="bch-compute", nodes=1, ntasks=1, cpus=16, mem="50G", time="24:00:00", sleep_time=5):
    
    
    """
    Submits a job to the Slurm job scheduler.

    This function creates a script that has the given command, and then submits this script to the Slurm job scheduler.
    
    Parameters:
    job_name (str): A name for the job. This will help you identify the job later.
    command (str): The command that you want to run.
    partition (str): The partition on the cluster where you want to run the job. Defaults to "bch-compute".
    nodes (int): The number of nodes (computers) that you want to use to run the job. Defaults to 1.
    cpus_per_task (int): The number of CPUs (processing units) that you want to use on each node. Defaults to 16.
    mem (str): The amount of memory that you want to use on each node. Defaults to "50GB".
    time (str): The maximum amount of time that the job is allowed to run. Defaults to "10:00:00" (10 hours).

    Returns:
    job_id (str): The ID of the job that was submitted. You can use this ID to check on the status of the job later.
    """

    
    script = f"""#!/bin/bash
    
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks} 
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH -o output_%j.txt
#SBATCH --mail-type=NONE

    # Run the command
    export MPLBACKEND=TkAgg
   
   source /lab-share/Neuro-Cohen-e2/Public/environment/load_neuroimaging_env.sh

    set -e

    {command}
    """
    
    if command == None:
        return
    else:
        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(script)
            script_file = f.name

        # Make the script executable
        subprocess.run(["chmod", "+x", script_file])

        try:
            # Submit the job using sbatch through the shell
            output = subprocess.check_output(['sbatch', script_file]).decode('utf-8')

            # Extract the job ID from the output
            job_id = output.strip().split()[-1]
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            job_id = None

    sleep(sleep_time)

    return job_id

def submit_slurm_job_test(job_name, command, partition="bch-compute", nodes=1, ntasks=1, cpus=16, mem="50G", time="24:00:00"):
    
    
    """
    Submits a job to the Slurm job scheduler.

    This function creates a script that has the given command, and then submits this script to the Slurm job scheduler.
    
    Parameters:
    job_name (str): A name for the job. This will help you identify the job later.
    command (str): The command that you want to run.
    partition (str): The partition on the cluster where you want to run the job. Defaults to "bch-compute".
    nodes (int): The number of nodes (computers) that you want to use to run the job. Defaults to 1.
    cpus_per_task (int): The number of CPUs (processing units) that you want to use on each node. Defaults to 16.
    mem (str): The amount of memory that you want to use on each node. Defaults to "50GB".
    time (str): The maximum amount of time that the job is allowed to run. Defaults to "10:00:00" (10 hours).

    Returns:
    job_id (str): The ID of the job that was submitted. You can use this ID to check on the status of the job later.
    """

    
    script = f"""#!/bin/bash
    
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}  
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH -o output_%j.txt
#SBATCH --mail-type=NONE

    # Run the command
    export MPLBACKEND=TkAgg
    
    source /lab-share/Neuro-Cohen-e2/Public/environment/load_neuroimaging_env.sh

    set -e

    {command}
    """
    
    # Write the script to a temporary file
    with open(f'slurm_script_{job_name}.sh', 'w') as f:
        f.write(script)

def set_registration_target(file_names, target1, target2):
    
    """
    Sets the registration target based on a list of file names.

    This function looks at the file names and sets the registration target to the first file that matches a certain criterion.

    Parameters:
    file_names (list): A list of file names.

    Returns:
    reg_target (str): The registration target.
    
    Note:
    The function checks if any of the file names contain the registration target strings (Reg_target_1 or Reg_target_2).
    If no registration target is found, a ValueError is raised.
    """
        

    reg_target = None
    for file_name in file_names:
        if target1 in file_name:
            reg_target = target1
            break
    if reg_target is None:
        for file_name in file_names:
            if target2 in file_name:
                reg_target = target2
                break
    if reg_target is None:
        raise ValueError(f"No registration target found in {file_names}")
    return reg_target


def combine_images(working_dir, list_of_images, out_name, clean_up=True):
#images should be inside working_dir somewhere 

#def combine_images(working_dir, input_dir, participant, session, image_type, list_of_images, clean_up=True):
    """
    Combines images of different directions using niftymic.
    
    Parameters:
    working_dir (str): The working directory.
    input_dir (str): The input directory.
    participant (str): The participant ID.
    session (str): The session ID.
    image_type (str): The image type.
    list_of_images (list): The list of images to combine.
    clean_up (bool): Whether to clean up temporary files. Default is True.
    
    Returns:
    command (str): The command to combine images.
    """

    for i, image in enumerate(list_of_images, start=1):
        mask_file = f'{working_dir}/temp_{i}_{out_name}_mask.nii.gz'
        

        result = subprocess.run(['fslmaths', image, '-abs', '-bin', mask_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(result.stderr.decode())
            raise Exception(f'Failed to create mask for {image}')
    
    mask_files = [f'{working_dir}/temp_{i}_{out_name}_mask.nii.gz' for i in range(1, len(list_of_images) + 1)]
    output_file = f'{working_dir}/{out_name}.nii.gz'
    
    
    cmd = [
        'singularity', 'exec',
        '-B', f'{working_dir}:/app/data',
        '-B', f'{script_dir}:{script_dir}',
        f'{script_dir}/niftymic.sif',
        'niftymic_reconstruct_volume',
        '--filenames', *list_of_images,
        '--filenames-masks', *mask_files,
        '--output', output_file
    ]
    
    if clean_up == True:
        cmd += '\n'
        cmd += [
            'rm', '-r', 
            f'{working_dir}/config*', 
            f'{working_dir}/temp*',  
            f'{working_dir}/motion_correction'
        ]
        
    command = ' '.join(cmd)

    return command

def reslice_image(input_image):
    """
    Reslices an image to 1mm isovolumetric if the largest pixel dimension is greater than 1.5mm.
    
    Parameters:
    input_folder (str): The input folder.
    participant (str): The participant ID.
    session (str): The session ID.
    reg_target (str): The registration target.
    """
    
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"File {input_image} does not exist")
    # Get the maximum pixel width
    cmd = f"fslinfo {input_image} | grep pixdim[1-3] | awk '{{ print $2 }}' | sort -rn | head -1"
    max_pixelwidth = float(subprocess.check_output(cmd, shell=True).strip())

    if max_pixelwidth > 1.5:
        print(f"Largest pixel dimension is {max_pixelwidth} > 1.5mm, reslicing to 1mm isovolumetric")
        
        stem = input_image.split('.')[0]
        size = 1
        output_file = f"{stem}_{size}mm.nii.gz"

        cmd = f"flirt -interp spline -in {input_image} -ref {input_image} -applyisoxfm {size} -out {output_file}"
        subprocess.run(cmd, shell=True)

        os.rename(input_image, f"{stem}_aniso.nii.gz")
        os.rename(output_file, input_image)
    else:
        print(f"Largest pixel dimension is {max_pixelwidth}, leaving image alone")
        

def bias_corr(input_image, image_type, skullstrip=None, clean_up=True):
    """
    Performs bias correction on an image.
    
    Parameters:
    input_folder (str): The input folder.
    participant (str): The participant ID.
    session (str): The session ID.
    reg_target (str): The registration target.
    mask (bool): Whether to create a brain mask. Default is True.
    clean_up (bool): Whether to clean up temporary files. Default is True.
    
    Returns:
    cmd (str): The command to run.
    """
        
    stem = input_image.split('.')[0] 
    folder = os.path.dirname(input_image)
    cmd="echo Starting\n"
    if os.path.exists(f"{stem}_orig.nii.gz"):
        print(f'{stem}_orig.nii.gz already exists, suggesting this image has been bias corrected already!')
        return
    
    # add scripts to Path so code can find them
    cmd = f"export PATH=$PATH:{script_dir}\n"
    
    # Run fsl_anat_alt.sh
    cmd += f"fsl_anat_alt.sh -i {stem} -t {image_type} --noreg --nosubcortseg --noseg\n"

    # Rename files
    cmd += f"mv {stem}.nii.gz {stem}_orig.nii.gz\n" 
    cmd += f"mv {stem}.anat/T1_biascorr.nii.gz {stem}.nii.gz\n" 

    
    if skullstrip in ['optibet', 'both']:
        suffix = '_optibet'
        cmd += f"mv {stem}.anat/{image_type}_biascorr_brain.nii.gz {stem}_SkullStripped{suffix}.nii.gz\n"
        cmd += f"mv {stem}.anat/{image_type}_biascorr_brain_mask.nii.gz {stem}_brain-mask{suffix}.nii.gz\n"
    
    if skullstrip in ['synthstrip', 'both']:
        suffix = '_synthstrip'
        out_file = f"{stem}_SkullStripped{suffix}.nii.gz"
        out_mask = f"{stem}_brain-mask{suffix}.nii.gz"
        cmd += f"mri_synthstrip -i {input_image} -o {out_file} -m {out_mask}\n"

        
    # Run fslmaths
    cmd += f"fslmaths {stem}.nii.gz {stem}.nii.gz -odt short\n"
    
    if clean_up == True:
        cmd += f"rm -r {stem}.anat\n"
        
    return cmd


def co_register(working_dir, target_image, moving_image, tag, brain_mask=None, clean_up=True):
    
    """
    Co-registers a moving image to a registration target.

    Parameters:
    working_dir (str): The working directory.
    reg_image (str): The registration target image.
    moving_image (str): The moving image.
    tag(str): Label to add to the filename of coregistered images
    skullstrip (bool): Whether to skullstrip the image. Default is True.
    clean_up (bool): Whether to clean up temporary files. Default is True.

    Returns:
    cmd (str): The command to run.
    """
    
    moving_stem=os.path.basename(moving_image).split('.')[0]
    target_stem=os.path.basename(target_image).split('.')[0]
    
    if os.path.exists(f"{working_dir}/{moving_stem}_{tag}.nii.gz"):
        print(f"WARNING: Input image file {moving_stem}_{tag}.nii.gz already exists. Skipping co-registration")
        return 
    
    if not os.path.exists(f'{working_dir}/warps_{moving_stem}'):
        os.makedirs(f'{working_dir}/warps_{moving_stem}')
    
   
    cmd = f"antsRegistrationSyNQuick.sh -d 3 -m {moving_image} -f {target_image} -t sr -o {working_dir}/warps_{moving_stem}/{moving_stem}_{tag}\n"
    
    cmd +=f"mv {working_dir}/warps_{moving_stem}/{moving_stem}_{tag}Warped.nii.gz {working_dir}/{moving_stem}_{tag}.nii.gz\n"
    
    if clean_up:
        cmd +=f" rm -r {working_dir}/warps_{moving_stem}\n"
    

    #Apply brain mask to other images to skull strip them
    if brain_mask:
        cmd += f"fslmaths {brain_mask} -mul {working_dir}/{moving_stem}_{tag}.nii.gz {working_dir}/{moving_stem}_{tag}_SkullStripped.nii.gz"
                        
    return cmd

def easy_reg(working_dir, source_brain, target_brain, tag, target_brain_seg=None, source_brain_seg=None, lesion_mask=None, other_brains=[], synthseg_robust=True):

    source_name=os.path.basename(source_brain).split('.')[0]
    target_name=os.path.basename(target_brain).split('.')[0]

    cmd = f'echo Running EasyReg for {source_brain}\n'
    cmd += "source activate easyreg\n"
    cmd += 'LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/openmpi3-gnu8/3.1.4/lib:/opt/ohpc/pub/compiler/gcc/8.3.0/lib64\n'
    cmd += 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))\n'
    cmd += 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:/lab-share/Neuro-Cohen-e2/Public/environment/conda/easyreg/lib/python3.9/site-packages/tensorrt_libs/:$LD_LIBRARY_PATH:\n'

    if synthseg_robust:
        cmd +=f"mri_synthseg --i {source_brain} --o {working_dir}/{source_name}_synthseg.nii.gz --parc --robust\n"
    
    if source_brain_seg == None:
        source_brain_seg=f'{working_dir}/{source_name}_synthseg.nii.gz'
    
    if target_brain_seg == None:
        target_brain_seg=f'{working_dir}/{target_name}_synthseg.nii.gz'
        
    cmd += ' '.join([
        'mri_easyreg',
        '--ref', target_brain,
        '--flo', source_brain,
        '--ref_seg', target_brain_seg,
        '--flo_seg', source_brain_seg,
        '--flo_reg', f'{working_dir}/{source_name}_{tag}.nii.gz',
        '--fwd_field', f'{working_dir}/{source_name}_{tag}Warp.nii.gz'
    ])
    
    cmd += "\n"
    
    if lesion_mask:
        cmd +=f"mri_easywarp --i {lesion_mask} --o {working_dir}/{source_name}_{tag}_lesion.nii.gz --field {working_dir}/{source_name}_{tag}Warp.nii.gz --nearest\n"
             
    if other_brains:
        for brain in other_brains:
            brain_name=os.path.basename(brain).split('.')[0]
            cmd +=f"mri_easywarp --i {brain} --o {working_dir}/{brain_name}_{tag}.nii.gz --field {working_dir}/{source_name}_{tag}Warp.nii.gz\n"

    return cmd
         
def ants_mni(working_dir, patient_brain, MNI_template, lesion_mask=None, other_brains=[], transform='s', histogram_matching=False, quick=False):
    
    """
    Co-registers a moving image to a registration target.

    Parameters:
    working_dir (str): The working directory.
    reg_image (str): The registration target image.
    moving_image (str): The moving image.
    skullstrip (bool): Whether to skullstrip the image. Default is True.
    clean_up (bool): Whether to clean up temporary files. Default is True.

    Returns:
    cmd (str): The command to run.
    """
    
    patient_stem=os.path.basename(patient_brain).split('.')[0]
    
    
    if not os.path.exists(f'{working_dir}/warps_{patient_stem}_space-MNI'):
        os.makedirs(f'{working_dir}/warps_{patient_stem}_space-MNI')
 
    
    if os.path.exists(f"{working_dir}/{patient_stem}_space-MNI.nii.gz"):
        print(f"WARNING: Input image file {patient_stem}_space-MNI.nii.gz already exists. Skipping Registration.")
        return 
    
    add_lesion_mask=''
    if lesion_mask:
        add_lesion_mask=f'-x {lesion_mask}'
    
    add_hist_match=''
    if histogram_matching == True:
        add_hist_match='-j 1'
    
    #The moving and fixed image are switched so that the lesion mask can be used in the registration
    # Usually the fixed image, aka target, would be the MNI brain 
    ants_cmd='antsRegistrationSyN.sh'
    if quick:
        ants_cmd='antsRegistrationSyNQuick.sh'
   
    cmd = f"{ants_cmd} -d 3 -m {MNI_template} -f {patient_brain} -t {transform} {add_lesion_mask} {add_hist_match} -o {working_dir}/warps_{patient_stem}_space-MNI/{patient_stem}_MNI\n"
    
    cmd +=f"mv {working_dir}/warps_{patient_stem}_space-MNI/{patient_stem}_MNIInverseWarped.nii.gz {working_dir}/{patient_stem}_space-MNI.nii.gz\n"
    
    if lesion_mask:
        lesion_stem=lesion_mask.split('.')[0]
        lesion_cmd = [
            'antsApplyTransforms', 
            '-d', '3', 
            '-i', f'{lesion_mask}', 
            '-r', f'{MNI_template}', 
            '-t', f'[{working_dir}/warps_{patient_stem}_space-MNI/{patient_stem}_MNI0GenericAffine.mat, 1]', 
            '-t', f'{working_dir}/warps_{patient_stem}_space-MNI/{patient_stem}_MNI1InverseWarp.nii.gz', 
            '-n', 'NearestNeighbor', 
            '-o', f'{working_dir}/{lesion_stem}_space-MNI.nii.gz\n'
        ]
        
        cmd += ' '.join(lesion_cmd) + '\n'

    if other_brains:
        for other_brain in other_brains:
            brain_stem=os.path.basename(other_brain).split('.')[0]
            other_brain_cmd = [
                'antsApplyTransforms', 
                '-d', '3', 
                '-i', f'{other_brain}', 
                '-r', f'{MNI_template}', 
                '-t', f'[{working_dir}/warps_{patient_stem}_space-MNI/{patient_stem}_MNI0GenericAffine.mat, 1]', 
                '-t', f'{working_dir}/warps_{patient_stem}_space-MNI/{patient_stem}_MNI1InverseWarp.nii.gz', 
                '-n', 'Linear', 
                '-o', f'{working_dir}/{brain_stem}_space-MNI.nii.gz\n'
            ]
        
            cmd += ' '.join(other_brain_cmd) + '\n'
    
                         
    return cmd

    

    
