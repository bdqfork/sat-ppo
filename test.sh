#!/bin/sh
main()
{
  python sat_eval.py -c muti -m model/sat/150/model-15099.pt -d data/satlab/300/test.txt -l 4 -f -a
  sleep 60
  python sat_eval.py -c net -m model/sat/150/model-15099.pt -d data/satlab/300/test.txt -l 4 -f -a
  sleep 60
  python sat_eval.py -c random -m model/sat/150/model-15099.pt -d data/satlab/300/test.txt -l 4 -f -a
  sleep 60
  python sat_eval.py -c lookahead -m model/sat/150/model-15099.pt -d data/satlab/300/test.txt -l 4 -f -a
}

main