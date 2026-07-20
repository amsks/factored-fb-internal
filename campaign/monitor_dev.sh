#!/usr/bin/env bash
# emit only on: new failures, or campaign completion (then exit).
S=/raid/tongle/rl/fb-data/campaign/logs_dev/status.json
prev=-1
while true; do
  read d r f fin <<<"$(python3 -c "import json;x=json.load(open('$S'));print(x['done'],x['running'],x['failed'],x['finished'])" 2>/dev/null || echo '0 0 0 False')"
  [ "$prev" = -1 ] && prev=$f
  if [ "$f" -gt "$prev" ]; then echo "DEV new failures: $f total ($d done, $r running)"; prev=$f; fi
  if [ "$fin" = "True" ]; then echo "DEV CAMPAIGN COMPLETE: $d done, $f failed of 50"; exit 0; fi
  sleep 120
done
