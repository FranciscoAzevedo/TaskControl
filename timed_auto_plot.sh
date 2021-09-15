# Shell script to run plots everyday for last session of given animals
declare -a animal_fd_paths=("/media/storage/shared-paton/georg/Animals_reaching/JJP-02630" 
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02633"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02385"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02396"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02398")

echo 'Changed Conda Env'
source /home/paco/anaconda3/etc/profile.d/conda.sh
conda activate TaskControl

# Loop through animals
echo 'Plotting daily figures'
for animal_fd_path in "${animal_fd_paths[@]}"
do
    echo "$animal_fd_path"
    # Get the path to the latest session
    python3 /media/paco/Data/TaskControl/daily_plot.py "$animal_fd_path"
done