#!/bin/sh

VIRTENV=$1

set -x
set -e

# This is the last version of virtualenv to support python 2.4:
curl https://raw.github.com/pypa/virtualenv/1.7.2/virtualenv.py
curl -O http://www.python.org/ftp/python/2.4.6/Python-2.4.6.tar.bz2
tar xjf Python-2.4.6.tar.bz2
cd Python-2.4.6
cat >setup.cfg <<EOF
[build_ext]
library_dirs=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)/
EOF
./configure --prefix=$PWD/install
make
make install
install/bin/python2.4 ./virtualenv.py --distribute $VIRTENV
