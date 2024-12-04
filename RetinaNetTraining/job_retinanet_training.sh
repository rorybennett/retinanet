#$ -pe smp 12            # Number of cores.
#$ -l h_vmem=7.5G        # Memory per core (max 11G per GPU).
#$ -l h_rt=240:0:0        # Requested runtime.
#$ -cwd                 # Change to current directory.
#$ -j y                 # Join output and error files.
#$ -o outputs/          # Change default output directory.
#$ -l gpu=1             # Request GPU usage.
#$ -t 1-6               # Array job.
#$ -tc 6                # Concurrent jobs.
#$ -m bea               # Email beginning, end, and aborted.

module load python

# Activate virtualenv
source venv/bin/activate

inputs=(
    "ProspectiveData/prostate_Combined/train_0"
    "ProspectiveData/prostate_Combined/val_0"
    "ProspectiveData/prostate_Combined/fold_0"
    "ProspectiveData/prostate_Combined/train_1"
    "ProspectiveData/prostate_Combined/val_1"
    "ProspectiveData/prostate_Combined/fold_1"
    "ProspectiveData/prostate_Combined/train_2"
    "ProspectiveData/prostate_Combined/val_2"
    "ProspectiveData/prostate_Combined/fold_2"
    "ProspectiveData/prostate_Combined/train_3"
    "ProspectiveData/prostate_Combined/val_3"
    "ProspectiveData/prostate_Combined/fold_3"
    "ProspectiveData/prostate_Combined/train_4"
    "ProspectiveData/prostate_Combined/val_4"
    "ProspectiveData/prostate_Combined/fold_4"
    "ProspectiveData/prostate_Combined/train_all"
    "ProspectiveData/prostate_Combined/val_all"
    "ProspectiveData/prostate_Combined/fold_all"
)


training_path=${inputs[$((3 * SGE_TASK_ID - 3))]}
validation_path=${inputs[$((3 * SGE_TASK_ID - 2))]}
saving_path=${inputs[$((3 * SGE_TASK_ID - 1))]}

python training_retinanet.py \
  --train_path="./Datasets/$training_path" \
  --val_path="./Datasets/$validation_path" \
  --save_path="/data/scratch/exx851/RetinaNetResults/$saving_path"
