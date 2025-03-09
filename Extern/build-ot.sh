#!/bin/bash
WORK_DIR=`pwd`
# BUILD_DIR=$WORK_DIR/build
# DEPS_DIR=$WORK_DIR/deps
if [ -d .git ]; then 
  git submodule init
  git submodule update
else
  git clone https://github.com/emp-toolkit/emp-tool.git $WORK_DIR/emp-tool
  git clone https://github.com/emp-toolkit/emp-ot.git $WORK_DIR/emp-ot
fi

cd emp-tool
git checkout 44b1dde
git apply ../patch/emp-tool.patch
cd ..

cd emp-ot
git checkout 0342af5
git apply ../patch/emp-ot.patch

# target=emp-tool
# cd $DEPS_DIR/$target
# git checkout 44b1dde
# patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/emp-tool.patch -d $DEPS_DIR/$target
# mkdir -p $BUILD_DIR/deps/$target
# cd $BUILD_DIR/deps/$target
# cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR
# make install -j2

# target=emp-ot
# cd $DEPS_DIR/$target
# git checkout 7f3d4f0
# mkdir -p $BUILD_DIR/deps/$target
# cd $BUILD_DIR/deps/$target
# cmake $DEPS_DIR/$target -DCMAKE_INSTALL_PREFIX=$BUILD_DIR -DCMAKE_PREFIX_PATH=$BUILD_DIR
# make install -j2