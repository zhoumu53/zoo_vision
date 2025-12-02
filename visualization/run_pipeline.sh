
date=20250901
# time in list: 02 06 10 14 18 22

for time in 02 06 10 14 18 22
do
    echo "Processing date: $date, time: $time"

    bash run_offline_stitching.sh $date $time

    echo "Running behavior analysis for date: $date, time: $time"

    # bash run_behavior_analysis.sh $date $time

done