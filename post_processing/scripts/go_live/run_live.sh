#!/usr/bin/env bash
set -euo pipefail

## run feature extraction 
CRON_LINE='30 4,5 * * * /bin/bash /media/mu/zoo_vision/post_processing/scripts/go_live/run_extraction.sh >> /media/mu/zoo_vision/post_processing/scripts/go_live/cron.log 2>&1'
( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -

CRON_LINE='0 6 * * * /bin/bash /media/mu/zoo_vision/post_processing/scripts/go_live/run_extraction.sh >> /media/mu/zoo_vision/post_processing/scripts/go_live/cron.log 2>&1'
( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -

## stitching - update db at 06:15, 07:15 daily
CRON_LINE='15 6 * * * /bin/bash /media/mu/zoo_vision/post_processing/scripts/go_live/run_post_processing_db_update.sh >> /media/mu/zoo_vision/post_processing/scripts/go_live/cron_2.log 2>&1'
( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -

## stitching
CRON_LINE='0 10 * * * /bin/bash /media/mu/zoo_vision/post_processing/scripts/go_live/run_post_processing_db_update.sh >> /media/mu/zoo_vision/post_processing/scripts/go_live/cron_2.log 2>&1'
( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -


CRON_LINE='0 11 * * * /bin/bash /media/mu/zoo_vision/post_processing/scripts/offline/offline_extraction_split3.sh >> /media/mu/zoo_vision/post_processing/scripts/offline/cron3.log 2>&1'
( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -