#!/bin/sh
#SBATCH --job-name=ljonk-shady-distributed
#SBATCH --account=bk-ur-uds
#SBATCH --partition=general

#SBATCH --qos=short  # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)

#SBATCH --time=0:30:00
#SBATCH --nodes=1                        # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=3              # This needs to match Fabric(devices=...)
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# Start measuring execution time
start_time=$(date +%s)

# The script uses the DATASETS_ROOT variable to locate the dataset.
export DATASETS_ROOT="/tudelft.net/staff-umbrella/UDSstorage/msc_ljonker/training_data/"
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/UDSstorage/msc_ljonker/containers/dev_image.sif

# Check that container file exists
if [ ! -f $APPTAINER_IMAGE ]; then
    ls $APPTAINER_IMAGE
    exit 1
fi

echo "Hostname: `hostname`"

# Load CUDA that is compatible to container libraries
module use /opt/insy/modulefiles
module load cuda/12.1

nvidia-smi

# Start GPU monitoring in the background
GPU_USAGE_FILE="logs/gpu-usage-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > $GPU_USAGE_FILE &
GPU_MONITOR_PID=$!

# Prevents direct GPU-to-GPU communication in NVIDIA's collective communications library
# Forcing data to flow through the CPU instead.
# export NCCL_P2P_DISABLE=1

# Run script
srun apptainer exec \
    --nv \
    -B /tudelft.net/:/tudelft.net/ \
    $APPTAINER_IMAGE \
    python train_validate.py --devices 1

# Terminate the GPU monitoring process
kill $GPU_MONITOR_PID

# End measuring execution time
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"