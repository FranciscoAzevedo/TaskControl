# Shell script to run plots everyday for last session of given animals
declare -a animal_fd_paths=("/media/storage/shared-paton/georg/Animals_reaching/JJP-02909"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02911"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02912"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02994"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02995"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02996"
                            "/media/storage/shared-paton/georg/Animals_reaching/JJP-02997")

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