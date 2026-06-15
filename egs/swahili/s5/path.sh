#!/usr/bin/env bash
export KALDI_ROOT="/data/git/kaldi"

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

# Add symbolic links to standard utils and steps if they don't exist
[ ! -L utils ] && ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
[ ! -L steps ] && ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
[ ! -L conf ] && ln -s $KALDI_ROOT/egs/wsj/s5/conf conf
[ ! -L shared ] && ln -s ../../../shared shared


export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
#export PATH=$PWD/utils/:$PWD/steps/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

DATA_DIR=$PWD/data
LEXICON=$DATA_DIR/local/dict/lexicon.txt
EXP_DIR="dev test"
TRAIN_DIR="train"


export LC_ALL=C
export PYTHONUNBUFFERED=1
