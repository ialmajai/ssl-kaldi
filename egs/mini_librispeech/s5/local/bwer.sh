#!/bin/bash

(
for x in exp*/*/decode*; do [ -d $x ] && [[ $x =~ "$1" ]] && grep WER $x/wer_* 2>/dev/null | local/best_wer.sh; done

for x in exp*/chain*/*/decode* ; do [ -d $x ] && [[ $x =~ "$1" ]] && grep WER $x/wer_* 2>/dev/null |  local/best_wer.sh; done
) | sort -n -k2
exit 0

