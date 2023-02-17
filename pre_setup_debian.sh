#!/bin/bash

sudo apt-get install -y wget

cwd=$(pwd)

script_dir=$(dirname "$0")

cd $script_dir

# install ilastik
wget https://files.ilastik.org/ilastik-1.4.0rc8-Linux.tar.bz2
tar xjf ilastik-1.*-Linux.tar.bz2
mv ilastik-1.*-Linux/* "$script_dir/ilastik/src/"
rm -rf ilastik-1.*-Linux
chmod +x "$script_dir/ilastik/src/run_ilastik.sh"

cd $cwd