#!/bin/bash

set -e

echo "$0 $@"
while getopts ":t:dz" opt; do
  case $opt in
    t)
      TARGET_SOC=$OPTARG
      ;;
    d)
      ENABLE_DMA32=ON
      export ENABLE_DMA32=TRUE
      ;;
    z)
      ENABLE_ZERO_COPY=ON
      export ENABLE_ZERO_COPY=TRUE
      ;;
    :)
      echo "Option -$OPTARG requires an argument." 
      exit 1
      ;;
    ?)
      echo "Invalid option: -$OPTARG index:$OPTIND"
      ;;
  esac
done

if [ -z ${TARGET_SOC} ] ; then
  echo "$0 -t <target> "
  echo ""
  echo "    -t : target (rk356x/rk3588)"
  echo "    -d : enable dma32"
  echo "    -z : enable zero copy"
  echo "such as: $0 -t rk3588 "
  echo ""
  exit -1
fi

case ${TARGET_SOC} in
    rk356x)
        ;;
    rk3588)
        ;;
    rk3576)
        TARGET_SOC="rk3576"
        ;;
    rk3566)
        TARGET_SOC="rk356x"
        ;;
    rk3568)
        TARGET_SOC="rk356x"
        ;;
    rk3562)
        TARGET_SOC="rk356x"
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3562,rk3566,rk3568,rk3576,rk3588"
        exit -1
        ;;
esac

# RGA2 only support under 4G memory
if [[ -z ${ENABLE_DMA32} ]];then
    ENABLE_DMA32=OFF
fi

# support zero copy api
if [[ -z ${ENABLE_ZERO_COPY} ]];then
    ENABLE_ZERO_COPY=OFF
fi

GCC_COMPILER=aarch64-linux-gnu
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
INSTALL_DIR=${ROOT_PWD}/install/${TARGET_SOC}_linux
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SOC}_linux

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "ENABLE_ALLOC_DMA32=${ENABLE_DMA32}"
echo "ENABLE_ZERO_COPY=${ENABLE_ZERO_COPY}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. \
    -DTARGET_SOC=${TARGET_SOC} \
    -DENABLE_DMA32=${ENABLE_DMA32} \
    -DENABLE_ZERO_COPY=${ENABLE_ZERO_COPY} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j4
make install

