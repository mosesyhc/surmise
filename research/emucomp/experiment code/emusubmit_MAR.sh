#!/bin/bash
while IFS=$'\t' read -r n function failrandom failfraction method rep
do
    JOB=`sbatch <<EOJ
#!/bin/bash
#SBATCH --account=p30845
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --mail-user=mosesyhc@u.northwestern.edu
#SBATCH -J emucompMAR
#SBATCH --output=error_output/R-%x.%j.out
#SBATCH --nodes=1
#SBATCH -n 1

# unload modules that may have been loaded when job was submitted
module purge all

# load the version of python you want to use
module load python/anaconda3.6
source activate surmise-venv

# By default all file paths are relative to the directory where you submitted the job.
python emucompare.py --n=${n} --function=${function} --failrandom=${failrandom} --failfraction=${failfraction} --method=${method} --rep=${rep}
EOJ
`

done < params0MAR.txt
exit

