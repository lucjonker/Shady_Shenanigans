#!/bin/sh
#SBATCH --job-name=ljonk-shady-make-csv
#SBATCH --account=bk-ur-uds
#SBATCH --partition=general

#SBATCH --qos=short  # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)

#SBATCH --time=0:03:00
#SBATCH --nodes=1                        # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=1              # This needs to match Fabric(devices=...)
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=logs/csvtask_%x_%j.out
#SBATCH --error=logs/csvtask_%x_%j.err

# The script uses the DATASETS_ROOT variable to locate the dataset.
export DATASETS_ROOT="/tudelft.net/staff-umbrella/UDSstorage/msc_ljonker/training_data/"
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/UDSstorage/msc_ljonker/containers/dev_image.sif

# Check that container file exists
if [ ! -f $APPTAINER_IMAGE ]; then
    ls $APPTAINER_IMAGE
    exit 1
fi

echo "Hostname: `hostname`"

# Run script
srun apptainer exec \
    -B /tudelft.net/:/tudelft.net/ \
    $APPTAINER_IMAGE \
    python main.py
