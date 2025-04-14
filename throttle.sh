#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

## replace DEV=lo with your card (e.g., eth0)
DEV=$2
if [ "$1" == "del" ]
then
	tc qdisc del dev $DEV root
fi
## lan in bumblebee
if [ "$1" == "lan" ]
then
tc qdisc del dev $DEV root
## about 3Gbps
tc qdisc add dev $DEV root handle 1: tbf rate 1000mbit burst 100000 limit 10000
## ping latency is double
tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.25msec
fi
## wan in bumblebee
if [ "$1" == "wan1" ]
then
sudo tc qdisc del dev $DEV root
## about 3Gbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
## about 0.3ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 2msec
fi

if [ "$1" == "wan2" ]
then
sudo tc qdisc del dev $DEV root
## about 400Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 800mbit burst 100000 limit 10000
## about 40ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 40msec
fi

if [ "$1" == "wan3" ]
then
sudo tc qdisc del dev $DEV root
## about 400Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 1600mbit burst 100000 limit 10000
## about 40ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 20msec
fi

if [ "$1" == "wan4" ]
then
sudo tc qdisc del dev $DEV root
## about 400Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 1600mbit burst 100000 limit 10000
## about 40ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 40msec
fi