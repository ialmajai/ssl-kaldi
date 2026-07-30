export KALDI_ROOT=${KALDI_ROOT:-/data/git/kaldi}

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

# Add symbolic links to standard utils and steps if they don't exist.
# Note: unlike the other recipes there is no "conf" symlink here, because TIMIT
# ships its own conf/ (speaker lists and the 60-48-39 phone map).
[ ! -L utils ] && ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
[ ! -L steps ] && ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
[ ! -L shared ] && ln -s ../../../shared shared

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1
