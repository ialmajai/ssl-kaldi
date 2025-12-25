# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

export train_cmd="run.pl --mem 2G"
export decode_cmd="run.pl --mem 4G"
export feats_cmd="run.pl --mem 4G"
#train_cmd="run.pl"
#decode_cmd="run.pl"
export cuda_cmd="queue.pl --gpu 1"
#cuda_cmd="run.pl"
# Do training locally.  Note: for jobs on smallish subsets,
# it's way faster to run on a single machine with a handful of CPUs, as
# you avoid the latency of starting GridEngine jobs.



