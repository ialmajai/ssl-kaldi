#!/usr/bin/env bash
# Print the best WER for every decode directory, sorted by WER.
#
# Run from a recipe directory:
#   shared/bwer.sh            # every decode dir
#   shared/bwer.sh chain      # only paths matching "chain"
#
# The optional argument is a substring matched against the decode directory
# path, so "tri3", "chain" or "dev" all work as filters.
#
# Globs two depths, covering exp/tri3b/decode and exp/chain/tdnn1a_sp/decode.
# Scoring itself is Kaldi's utils/best_wer.sh.

(
  for x in exp*/*/decode*; do
    [ -d "$x" ] && [[ $x =~ "${1:-}" ]] && \
      grep WER "$x"/wer_* 2>/dev/null | utils/best_wer.sh
  done

  for x in exp*/*/*/decode*; do
    [ -d "$x" ] && [[ $x =~ "${1:-}" ]] && \
      grep WER "$x"/wer_* 2>/dev/null | utils/best_wer.sh
  done
) | sort -n -k2
exit 0
