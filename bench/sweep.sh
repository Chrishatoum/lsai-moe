#!/bin/bash
# bench/sweep.sh [LEVER] [NODES...]   e.g. bench/sweep.sh baseline 1 2 4 8
cd "$(dirname "$0")/.."
lever=${1:-baseline}; shift
nodes=${@:-1 2 4 8}
declare -A experts=([g1]=8 [g2]=16 [g4]=32 [g8]=64 [g16]=128)
for n in $nodes; do
  for g in dense g1 g2 g4 g8 g16; do
    [ "$g" != dense ] && [ $((n * 4)) -gt ${experts[$g]} ] && continue
    bench/submit.sh train $g $lever $n
  done
done
