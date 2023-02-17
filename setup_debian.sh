#!/bin/bash

sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

curl https://pyenv.run | bash

echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

cwd=$(pwd)

script_dir=$(dirname "$0")

cd $script_dir


# install ilastik
wget https://files.ilastik.org/ilastik-1.4.0rc8-Linux.tar.bz2
tar xjf ilastik-1.*-Linux.tar.bz2
mv ilastik-1.*-Linux/* "$script_dir/ilastik/src/"
rm -rf ilastik-1.*-Linux
chmod +x "$script_dir/ilastik/src/run_ilastik.sh"


# Install ants
cd "$script_dir/brat_tissue/module/lib/ants"
bash install.sh

cd $cwd