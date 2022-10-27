#!/bin/bash
IDS=$(sbatch submit_restart.job)
ID=${IDS//[!0-9]/}
echo "ID $ID"

for t in {1..7..1}
do
  IDS=$(sbatch --dependency=afterany:$ID submit_restart.job)
  ID=${IDS//[!0-9]/}
  echo "ID $ID"
done

