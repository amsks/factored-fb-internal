#!/usr/bin/env bash
# campaign/monitor_campaign.sh — emit an event only on: new failures, native-complete,
# or full-campaign-complete (then exit). Quiet otherwise. Drives a Monitor watch.
C=/raid/tongle/rl/fb-data/campaign
read_status() {  # $1=status.json -> "done running failed finished"
    python3 -c "import json;d=json.load(open('$1'));print(d['done'],d['running'],d['failed'],d['finished'])" 2>/dev/null || echo "0 0 0 False"
}
prev_nf=-1; prev_jf=-1; native_announced=0
while true; do
    read nd nr nf nfin <<<"$(read_status $C/logs_native/status.json)"
    [ "$prev_nf" = -1 ] && prev_nf=$nf
    if [ "$nf" -gt "$prev_nf" ]; then echo "NATIVE new failures: total $nf failed ($nd done, $nr running)"; prev_nf=$nf; fi
    if [ "$nfin" = "True" ] && [ "$native_announced" = 0 ]; then echo "NATIVE COMPLETE: $nd done, $nf failed of 114"; native_announced=1; fi
    if [ -f $C/logs_jax/status.json ]; then
        read jd jr jf jfin <<<"$(read_status $C/logs_jax/status.json)"
        [ "$prev_jf" = -1 ] && prev_jf=$jf
        if [ "$jf" -gt "$prev_jf" ]; then echo "JAX new failures: total $jf failed ($jd done, $jr running)"; prev_jf=$jf; fi
        if [ "$jfin" = "True" ]; then echo "CAMPAIGN COMPLETE: native ${nd}done/${nf}fail + jax ${jd}done/${jf}fail"; exit 0; fi
    fi
    sleep 120
done
