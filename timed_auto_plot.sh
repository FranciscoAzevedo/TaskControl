# Shell script to run plots everyday for last session of given animals
declare -a animal_fd_paths=("/media/storage/shared-paton/georg/Animals_reaching/JJP-02912_Teacher"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02995_Poolboy"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02997_Therapist")

echo 'Changed Conda Env'
source /home/paco/anaconda3/etc/profile.d/conda.sh
conda activate TaskControl

# Loop through animals
echo 'Plotting daily figures'
for animal_fd_path in "${animal_fd_paths[@]}"
do
    echo "Current animal: $animal_fd_path "
    # Get the path to the latest session
    python3 /media/paco/Data/TaskControl/daily_plot.py "$animal_fd_path"
done