#!/bin/bash

BASEDIR=$(dirname "$0")

if [ "$(grep -Ei 'debian|buntu|mint' /etc/*release)" ]; then
   bash "${BASEDIR}/pre_setup_debian.sh"
fi