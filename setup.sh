#!/bin/bash

BASEDIR=$(dirname "$0")

if [ "$(grep -Ei 'debian|buntu|mint' /etc/*release)" ]; then
   bash "${BASEDIR}/setup_debian.sh"
fi