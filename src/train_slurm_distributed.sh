#!/bin/sh
#SBATCH --job-name=ljonk-shady-4gpu
#SBATCH --partition=general
#SBATCH --account=ewi-insy-reit

##SBATCH --qos=short
#SBATCH --qos=reservation
#SBATCH --reservation=reit-course-scalable-ai

#SBATCH --time=0:02:00
#SBATCH --nodes=1                        # This needs to match Fabric(num_nodes=...)
#SBATCH --ntasks-per-node=4              # This needs to match Fabric(devices=...)
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=1G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# Start measuring execution time
start_time=$(date +%s)

export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/REITcourses/apptainer/pytorch2.2.1-cuda12.1.sif

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
GPU_USAGE_FILE="gpu-usage-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > $GPU_USAGE_FILE &
GPU_MONITOR_PID=$!

# Prevents direct GPU-to-GPU communication in NVIDIA's collective communications library
# Forcing data to flow through the CPU instead.
# export NCCL_P2P_DISABLE=1

# Run script
srun apptainer exec \
    --nv \
    --env-file /tudelft.net/staff-umbrella/REITcourses/.env \
    -B /home/:/home/ \
    -B /tudelft.net/:/tudelft.net/ \
    $APPTAINER_IMAGE \
    python script.py --devices 4

# Terminate the GPU monitoring process
kill $GPU_MONITOR_PID

# End measuring execution time
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"