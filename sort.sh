#!/bin/sh
main()  
{  
  input=$5
  output=$6
  parallel=$1
  sep=$4
  limit=$3
  index=$2
  reverse=$7
  split -d -l $limit $input $input.part.
  parts=$(ls $input.part.*)
  cmd="sort --parallel=$parallel -t $sep -nrk$index"
  if [ ! -n "$reverse" ] ;then
    cmd="sort --parallel=$parallel -t $sep -nk$index"
  fi
  for part in $parts
  do
    $cmd $part > $part.sorted
  done
  sort -m $input.part.*.sorted > $output
  rm -f $input.part.*
}  
main $1 $2 $3 $4 $5 $6 $7