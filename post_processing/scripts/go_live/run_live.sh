#!/usr/bin/env bash
set -euo pipefail

### run feature extraction at 04:00, 06:00 daily
CRON_LINE='0 2,4,6 * * * /bin/bash /home/dherrera/git/zoo_vision/post_processing/scripts/go_live/run_extraction.sh >> /home/dherrera/git/zoo_vision/post_processing/scripts/go_live/cron.log 2>&1'

( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -


## stitching - update db at 06:15 daily
CRON_LINE='15 6 * * * /bin/bash /home/dherrera/git/zoo_vision/post_processing/scripts/go_live/run_post_processing_db_update.sh >> /home/dherrera/git/zoo_vision/post_processing/scripts/go_live/cron_2.log 2>&1'

( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -