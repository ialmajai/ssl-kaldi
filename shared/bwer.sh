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
# Recipes nest their experiment directories differently: exp/tri3b/decode in
# most, exp/chain/tdnn1a_sp/decode for chain models, and exp/<x>/<y>/decode in
# swahili. Both depths are globbed here, which is why this file replaces the
# per-recipe copies that each handled only their own layout.
#
# Scoring itself is Kaldi's utils/best_wer.sh. The recipes used to carry a
# byte-identical copy at local/best_wer.sh; those were removed as redundant.

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
