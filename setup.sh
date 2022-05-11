#!/bin/bash

if [ "$(grep -Ei 'debian|buntu|mint' /etc/*release)" ]; then
   bash setup_debian.sh
fi