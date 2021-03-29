#!/bin/bash
# Find our package manager

sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update

sudo apt-get install git
sudo apt install gcc
sudo apt install g++
sudo apt-get install build-essential
sudo apt-get install postgresql
sudo apt install daemonize
sudo apt-get install libpq-dev
sudo apt-get install net-tools



mkdir sw
cd sw
if grep -q microsoft /proc/version; then
  #Install SYSTEMCTL
  echo "Install WSL Stuff... systemctl"
  git clone https://github.com/DamionGans/ubuntu-wsl2-systemd-script.git
  bash ubuntu-wsl2-systemd-script/ubuntu-wsl2-systemd-script.sh
else
  echo "native Linux"
fi

cd ~/

# echo 'Installing Python3'
# sudo apt-get install python3 python3-pip python3-dev python3-setuptools
# python3 -m pip install --upgrade pip
# pip3 install virtualenv


echo 'Installing Anaconda Python (follow instructions, agree & yes^10)'
if [ -z "$CONDA_EXE" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ./Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc

    ~/miniconda3/bin/conda create -n py3
    ~/miniconda3/bin/conda activate py3
    ~/miniconda3/bin/conda install -c anaconda pip
    ~/miniconda3/bin/pip install -U pip-tools
fi



if [ -f "~/.ssh/id_rsa" ]; then
    echo 'Setting Up Github Account (first input)'
    ssh-keygen -t rsa -b 4096 -C "$1"
    git config --global user.email "$1"
    eval $(ssh-agent -s)
    ssh-add ~/.ssh/id_rsa

    echo 'Add your public key to your github account'
    cat < ./.ssh/id_rsa.pub
fi

read -p "Press enter to continue"

echo 'Installing Ottermatics Lib'
git clone git@github.com:SoundsSerious/ottermaticslib.git
cd ottermaticslib
~/miniconda3/bin/python3 -m pip install -r requirements.txt
~/miniconda3/bin/python3 setup.py install


echo "export DISPLAY=:0" >> ~/.bashrc 
sudo apt-get install x11-apps
sudo apt-get install libxkbcommon-x11-0
sudo apt-get install qt5-default
sudo apt-get install mesa-common-dev

alias ipythonqt="ipython qtconsole --style=monokai --gui-completion=droplist"
