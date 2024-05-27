# Image Processing Pipeline

This pipeline is designed to process images for further analysis. The pipeline consists of several steps:

1. Create Input Dictionary: Creates a dictionary of subjects and their corresponding sessions from a given input folder.
2. Reslice Image: Reslices an image to 1mm isovolumetric if the largest pixel dimension is greater than 1.5mm.
3. Bias Correction: Performs bias correction on an image using FSL.
4. Co-Register: Co-registers a moving image to a registration target using ANTs.

Usage

Create an input dictionary by running create_input_dict(input_dir, input_type).
Reslice images by running reslice_image(input_folder, participant, session, reg_target).
Perform bias correction by running bias_corr(input_folder, participant, session, reg_target, mask=True, clean_up=True).
Co-register images by running co_register(working_dir, reg_image, moving_image, skullstrip=True, clean_up=True).

Functions

print_tree(d, n=5, indent=0): Recursively prints the folder structure.
combine_images(working_dir, input_dir, participant, session, image_type, list_of_images, clean_up=True): Combines multiple images into a single image.
reslice_image(input_folder, participant, session, reg_target): Reslices an image to 1mm isovolumetric if the largest pixel dimension is greater than 1.5mm.
bias_corr(input_folder, participant, session, reg_target, mask=True, clean_up=True): Performs bias correction on an image.
co_register(working_dir, reg_image, moving_image, skullstrip=True, clean_up=True): Co-registers a moving image to a registration target.
Usage

Create a folder structure dictionary using print_tree.
Combine images using combine_images.
Reslice images using reslice_image.
Perform bias correction using bias_corr.
Co-register images using co_register.
