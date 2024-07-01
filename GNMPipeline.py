import os
import shutil
from glob import glob
from nilearn import plotting
import nibabel as nib
import subprocess
import sys
import matplotlib_inline
from tqdm import tqdm
import tempfile
from pathlib import Path
from time import sleep
import numpy as np

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
    If input_type is 'BIDS', the function assumes that the input folder is organized with subject folders containing session folders.
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
            subject_path = os.path.join(input_folder, subject)

            sessions = [f for f in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, f))]

            subject_sessions[subject] = sessions


        print(subject_sessions)
        
    
    elif input_type=='Folder':
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
 
def submit_slurm_job(job_name, command, partition="bch-compute", nodes=1, ntasks=1, cpus=16, mem="32G", time="10:00:00", sleep_time=5):
    
    
    """
    Submits a job to the Slurm job scheduler.

    This function creates a script that has the given command, and then submits this script to the Slurm job scheduler.
    
    Parameters:
    job_name : str
        A name for the job. This will help you identify the job later.
    command : str
        The command that you want to run.
    partition : str, optional
        The partition on the cluster where you want to run the job. Defaults to "bch-compute".
    nodes : int, optional
        The number of nodes (computers) that you want to use to run the job. Defaults to 1.
    ntasks : int, optional
        The number of tasks (or processes) to use for the computation. Default is 1.
    cpus_per_task : int, optional
        The number of CPUs (processing units) that you want to use on each node. Defaults to 16.
    mem : str, optional
        The amount of memory that you want to use on each node. Defaults to "32G".
    time : str, optional
        The maximum amount of time that the job is allowed to run. Defaults to "10:00:00" (10 hours).
    sleep_time : int, optional
        The number of seconds the function will wait (sleep) before returning. Defaults to 5. 

    Returns:
    job_id : str
        The ID of the job that was submitted. You can use this ID to check on the status of the job later.
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
    Creates an SBATCH file, but does not submit it to SLURM

    This function creates and saves a script that has the given command.
    
    Parameters:
    ----------
    job_name : str
        A name for the job. This will help you identify the job later.
    command : str
        The command that you want to run.
    partition : str, optional
        The partition on the cluster where you want to run the job. Defaults to "bch-compute".
    nodes : int, optional
        The number of nodes (computers) that you want to use to run the job. Defaults to 1.
    ntasks : int, optional
        The number of tasks (or processes) to use for the computation. Defaults to 1.
    cpus : int, optional
        The number of CPUs (processing units) that you want to use on each node. Defaults to 16.
    mem : str, optional
        The amount of memory that you want to use on each node. Defaults to "50G".
    time : str, optional
        The maximum amount of time that the job is allowed to run. Defaults to "24:00:00" (24 hours).

    Returns:
    -------
    None
    
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
    target1 (str): Ideal file type for registration. Case sensitive. 
    target2 (str): Back-up file type for registration if first not avaiable. Case sensitive. 

    Returns:
    reg_target (str): The available registration target file type. 
    
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

    """
    Generates a command to combine images of different directions using niftymic.
    
    Parameters:
    working_dir (str): The working directory.
    list_of_images (list of Nifti-like objects): The list of images to combine.
    out_name (str): The name of the output image. 
    clean_up (bool): Whether to clean up temporary files. Default is True.
    
    Returns:
    command (str): The command to combine images.
    """

    mask_files=[]
    for i, image in enumerate(list_of_images, start=1):
        image_name=os.path.basename(image)
        mask_file = f'{working_dir}/temp_{i}_{image_name}_mask.nii.gz'
        mask_files.append(mask_file)
        
        result = subprocess.run(['fslmaths', image, '-abs', '-bin', mask_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(result.stderr.decode())
            raise Exception(f'Failed to create mask for {image}')
    
    #mask_files = [f'{working_dir}/temp_{i}_{out_name}_mask.nii.gz' for i in range(1, len(list_of_images) + 1)]
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
    Reslices an image to 1mm isovolumetric if the largest pixel dimension is greater than 1.5mm
    
    Parameters:
    input_image (path to Nifti-like object): The input image to be resliced.

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
        

def bias_corr(input_image, image_type, skullstrip='synthstrip', clean_up=True):
    """
    Generates a command to bias correct, crop, and reorient an image using fsl_anat_alt.sh
    
    Parameters:
    input_image (path to Nifti-like object): Image to Bias correct. 
    image_type (str): Should be T1w or T2w for bias correction.
    skullstrip (str): Software to use for skull stripping. Should be 'optibet', 'synthstrip', 'both' or None. Defaults to 'synthstrip'. 
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


def co_register(working_dir, target_image, moving_image, tag="", brain_mask=None, clean_up=True):
    
    """
    Generates a command to co-register a moving image to a registration target.

    Parameters:
    working_dir (str): The working directory.
    target_image (path to Nifti-like object): The registration target image.
    moving_image (path to Nifti-like object): The moving image.
    tag (str): Label to add to the end of the name of outputs.
    brain_mask (path to Nifti-like object): Path to the brain mask of the target image to use to skull strip co-registed images. Defulat is None.
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

def synthseg_wrapper(input_list,output_list=[],robust=True, clean_up=False):
    
    """
    Generates a command to run synthseg on a list of input images. Faster than instantiating individual calls.
    
    Parameters:
    input_list (list of Nifti-like objects): Paths to files to segment.
    output_list (list of output files paths, optional): Paths the output segmentations will be saved. If none provided, appends '_synthseg' to input name. 
    
    Returns:
    command (str): The command to run.
    """
    
    if os.path.exists("synthseg_inputs.txt") or os.path.exists("synthseg_outputs.txt"):
        print(f"WARNING:  synthseg_inputs.txt and/or synthseg_outputs.txt already exists.")
        raise FileExistsError("Existing files detected")
    
    if not output_list:
        output_list=[i.split('.')[0]+'_synthseg.nii.gz' for i in input_list]
    
    with open("synthseg_inputs.txt", "w") as file:
            for item in input_list:
                file.write(f"{item}\n")
    with open("synthseg_outputs.txt", "w") as file:
        for item in output_list:
            file.write(f"{item}\n")
    
    cmd =[
    "mri_synthseg",
    "--i", "synthseg_inputs.txt", 
    "--o", "synthseg_outputs.txt",
    "--parc"
    ]
    
    if robust:
        cmb.append("--robust")
    
    if clean_up:
        cmd += "rm synthseg_inputs.txt synthseg_outputs.txt"
        
    command=" ".join(cmd)
    return command

def easy_reg(working_dir, source_brain, target_brain, target_brain_seg=None, source_brain_seg=None, lesion_mask=None, tag="", other_brains=[], synthseg_robust=False, affine=False):
    
    """
    Generates a command to run the EasyReg pipeline for brain image registration and optional lesion masking.

    Parameters:
    working_dir (str): Directory where output files will be saved.
    source_brain (str): Path to the source brain image.
    target_brain (str): Path to the target brain image.
    target_brain_seg (str, optional): Path to the target brain segmentation. Default is None.
    source_brain_seg (str, optional): Path to the source brain segmentation. Default is None.
    lesion_mask (str, optional): Path to the lesion mask. Default is None.
    tag (str, optional): Additional tag for output file naming. Default is an empty string.
    other_brains (list, optional): List of other brain images to be registered to the target. Default is an empty list.
    synthseg_robust (bool, optional): Whether to use the robust mode for SynthSeg. Default is False.
    affine (bool, optional): Whether to run affine-only registration. Default is False

    Returns:
    str: The command to run the EasyReg pipeline.
    """

    source_name=os.path.basename(source_brain).split('.')[0]
    target_name=os.path.basename(target_brain).split('.')[0]
    out_name=f'{source_name}_to_{target_name}{tag}'
    
    if source_brain_seg is None:
        source_brain_seg = f'{working_dir}/{source_name}_synthseg.nii.gz'

    if target_brain_seg is None:
        target_brain_seg = f'{working_dir}/{target_name}_synthseg.nii.gz'

    cmd = [
        f'echo Running EasyReg for {source_brain}',
        'source activate easyreg',
        'LD_LIBRARY_PATH=/opt/ohpc/pub/mpi/openmpi3-gnu8/3.1.4/lib:/opt/ohpc/pub/compiler/gcc/8.3.0/lib64',
        'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))',
        'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:/lab-share/Neuro-Cohen-e2/Public/environment/conda/easyreg/lib/python3.9/site-packages/tensorrt_libs/:$LD_LIBRARY_PATH:\n'
    ]

    if synthseg_robust:
        cmd.append(f"mri_synthseg --i {source_brain} --o {source_brain_seg} --parc --robust\n")

        
    mri_easyreg_cmd = [
        'mri_easyreg',
        '--ref', target_brain,
        '--flo', source_brain,
        '--ref_seg', target_brain_seg,
        '--flo_seg', source_brain_seg,
        '--flo_reg', f'{working_dir}/{out_name}.nii.gz',
        '--fwd_field', f'{working_dir}/{out_name}Warp.nii.gz',
        '--threads', '-1'
    ]

    if affine:
        mri_easyreg_cmd.append('--affine_only')
    
    cmd.append(' '.join(mri_easyreg_cmd) + '\n')
    
    if lesion_mask:
        cmd.append(f"mri_easywarp --i {lesion_mask} --o {working_dir}/{out_name}_lesion.nii.gz --field {working_dir}/{out_name}Warp.nii.gz --nearest\n")

    if other_brains:
        for brain in other_brains:
            brain_name = os.path.basename(brain).split('.')[0]
            cmd.append(f"mri_easywarp --i {brain} --o {working_dir}/{brain_name}_to_{target_name}{tag}.nii.gz --field {working_dir}/{out_name}Warp.nii.gz\n")

    cmd.append(f"mkdir {working_dir}/warps_{out_name}\n")
    cmd.append(f"mv {working_dir}/{out_name}Warp.nii.gz {working_dir}/warps_{out_name}\n")
    cmd.append(f"mv {source_brain_seg} {working_dir}/warps_{out_name}\n")
    
    if os.path.exists(f'{working_dir}/{target_name}_synthseg.nii.gz'):
        cmd +=f"mv f'{working_dir}/{target_name}_synthseg.nii.gz' {working_dir}/warps_{out_name}"
    
    return "\n".join(cmd)
         
def ants_mni(working_dir, patient_brain, MNI_template, lesion_mask=None, other_brains=[], tag="", transform='s', histogram_matching=False, quick=False):
    
    """
    Generates a command to run ANTs registration of a patient brain to an MNI template, with optional lesion masking and additional brain images.

    Parameters:
    working_dir (str): Directory where output files will be saved.
    patient_brain (str): Path to the patient brain image.
    MNI_template (str): Path to the MNI template image.
    lesion_mask (str, optional): Path to the lesion mask. Default is None.
    other_brains (list, optional): List of other brain images to be warped to the MNI template. Default is an empty list.
    tag (str, optional): Additional tag for output file naming. Default is an empty string.
    transform (str, optional): Transformation type. Default is 's'.
        's' for rigid + affine + deformable syn (3 stages)
        'b' for rigid + affine + deformable b-spline syn (3 stages)
        'a' for rigid + affine (2 stages)
    histogram_matching (bool, optional): Whether to use histogram matching. Default is False.
    quick (bool, optional): Whether to use the quick version of ANTs registration. Default is False.

    Returns:
    str: The command to run the ANTs registration pipeline.
    """

    patient_stem=os.path.basename(patient_brain).split('.')[0]
    
    
    if not os.path.exists(f'{working_dir}/warps_{patient_stem}_to_MNI{tag}'):
        os.makedirs(f'{working_dir}/warps_{patient_stem}_to_MNI{tag}')
 
    
    if os.path.exists(f"{working_dir}/{patient_stem}_to_MNI{tag}.nii.gz"):
        print(f"WARNING: Input image file {patient_stem}_to_MNI{tag}.nii.gz already exists. Skipping Registration.")
        return 
    
   
    ants_cmd = ["antsRegistrationSyNQuick.sh" if quick else "antsRegistrationSyN.sh"]

    ants_cmd += [
        "-d", "3",
        "-m", str(MNI_template),
        "-f", str(patient_brain),
        "-t", str(transform),
        "-o", f"{working_dir}/warps_{patient_stem}_space-MNI/{patient_stem}_MNI{tag}"
    ]

    if lesion_mask:
        ants_cmd.append("-x")
        ants_cmd.append(str(lesion_mask))

    if histogram_matching:
        ants_cmd.append("-j")
        ants_cmd.append("1")

    
    mv_cmd=["mv",
            f"{working_dir}/warps_{patient_stem}_to_MNI{tag}/{patient_stem}_to_MNI{tag}InverseWarped.nii.gz",
            f"{working_dir}/{patient_stem}_to_MNI{tag}.nii.gz"
           ]
    

    command_string = " && ".join([" ".join(ants_cmd), " ".join(mv_cmd)])
    
    if lesion_mask:
        #lesion_stem=os.path.basename(lesion_mask).split('.')[0]
        lesion_cmd = [
            "antsApplyTransforms", 
            "-d", "3", 
            "-i", str(lesion_mask), 
            "-r", str(MNI_template), 
            "-t", f"[{working_dir}/warps_{patient_stem}_to_MNI{tag}/{patient_stem}_to_MNI{tag}0GenericAffine.mat, 1]", 
            "-t", f"{working_dir}/warps_{patient_stem}_to_MNI{tag}/{patient_stem}_to_MNI{tag}1InverseWarp.nii.gz", 
            "-n", "NearestNeighbor", 
            "-o", f"{working_dir}/{patient_stem}_to_MNI{tag}_lesion.nii.gz\n"
        ]
        
        command_string += "\n\n" + " ".join(lesion_cmd)
    

    if other_brains:
        for other_brain in other_brains:
            brain_stem=os.path.basename(other_brain).split('.')[0]
            #brain_stem=os.path.basename(other_brain).split('_space')[0] 
            other_brain_cmd = [
                "antsApplyTransforms", 
                "-d", "3", 
                "-i", f"{other_brain}", 
                "-r", f"{MNI_template}", 
                "-t", f"[{working_dir}/warps_{patient_stem}_to_MNI/{patient_stem}_to_MNI{tag}0GenericAffine.mat, 1]", 
                "-t", f"{working_dir}/warps_{patient_stem}_to_MNI/{patient_stem}_t0_MNI{tag}1InverseWarp.nii.gz", 
                "-n", "Linear", 
                "-o", f"{working_dir}/{brain_stem}_to_MNI{tag}.nii.gz\n"
            ]
        
            command_string += "\n\n" + " ".join(other_brain_cmd)
              
    return command_string

class BrainImageMetrics:
    
    """
    A class to compute similarity metrics between two brain images.

    Attributes:
    -----------
    img1 : nib.Nifti1Image
        The first brain image loaded using nibabel.
    img2 : nib.Nifti1Image
        The second brain image loaded using nibabel.
    mask1 : np.ndarray
        Binary mask of the first brain image, where non-zero voxels are set to 1.
    mask2 : np.ndarray
        Binary mask of the second brain image, where non-zero voxels are set to 1.

    Methods:
    --------
    compute_dice_coefficient():
        Computes the Dice similarity coefficient between the two brain images.

    compute_jaccard_coefficient():
        Computes the Jaccard similarity coefficient between the two brain images.

    compute_non_overlapping_percentage():
        Computes the percentage of non-overlapping voxels between the two brain images.
    
    Example Usage:
    --------------
    metrics_calculator = BrainImageMetrics(brain_path1, brain_path2)
    
    dice_coefficient, jaccard_coefficient, non_overlapping_percentage = metrics_calculator.compute_metrics()
    dice_coefficient = metrics_calculator.compute_dice_coefficient()
    
    """
    
    def __init__(self, brain_path1, brain_path2):
        
        """
        Initializes the BrainImageMetrics class with two brain image paths.

        Parameters:
        -----------
        brain_path1 : str to Nifti-like object
            File path to the first brain image.
        brain_path2 : str to Nifti-like object
            File path to the second brain image.
        """
        self.img1 = nib.load(brain_path1)
        self.img2 = nib.load(brain_path2)
        self.mask1 = (np.array(self.img1.dataobj) > 0).astype(int)
        self.mask2 = (np.array(self.img2.dataobj) > 0).astype(int)
    
    def compute_dice_coefficient(self):
        """
        Computes the Dice coefficient between the two brain masks.
        
        Returns:
        -------
        float
            The Dice coefficient. Returns NaN if both masks are empty.
        """
        volume_sum = self.mask1.sum() + self.mask2.sum()
        if volume_sum == 0:
            return np.NaN
        else:
            volume_intersect = (self.mask1 & self.mask2).sum()
            return 2 * volume_intersect / volume_sum
    
    def compute_jaccard_coefficient(self):
        """
        Computes the Jaccard coefficient between the two brain masks.
        
        Returns:
        -------
        float
            The Jaccard coefficient. Returns NaN if the union of both masks is empty.
        """
        volume_intersect = (self.mask1 & self.mask2).sum()
        volume_union = (self.mask1 | self.mask2).sum()
        if volume_union == 0:
            return np.NaN
        else:
            return volume_intersect / volume_union
    
    def compute_non_overlapping_percentage(self):
        """
        Computes the percentage of non-overlapping voxels between the two brain masks.
        
        Returns:
        -------
        float
            The percentage of non-overlapping voxels. Returns NaN if both masks are empty.
        """
        total_voxels_mask1 = self.mask1.sum()
        total_voxels_mask2 = self.mask2.sum()
        overlapping_voxels = (self.mask1 & self.mask2).sum()
        non_overlapping_voxels_mask1 = total_voxels_mask1 - overlapping_voxels
        non_overlapping_voxels_mask2 = total_voxels_mask2 - overlapping_voxels
        total_non_overlapping_voxels = non_overlapping_voxels_mask1 + non_overlapping_voxels_mask2
        total_voxels = total_voxels_mask1 + total_voxels_mask2
        if total_voxels == 0:
            return np.NaN
        else:
            return (total_non_overlapping_voxels / total_voxels) * 100    
    
    def compute_metrics(self):
        """
        Computes the Dice coefficient, Jaccard coefficient, and non-overlapping percentage between the two brain masks.
        
        Returns:
        -------
        tuple
            A tuple containing the Dice coefficient, Jaccard coefficient, and non-overlapping percentage.
        """
        dice_coefficient = self.compute_dice_coefficient()
        jaccard_coefficient = self.compute_jaccard_coefficient()
        non_overlapping_percentage = self.compute_non_overlapping_percentage()
        return dice_coefficient, jaccard_coefficient, non_overlapping_percentage

    
