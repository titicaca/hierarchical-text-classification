#!/bin/sh

set -x

PROJECT_HOME=.

wget http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz -O /tmp/reuters21578.tar.gz
mkdir $PROJECT_HOME/data
tar -zxvf /tmp/reuters21578.tar.gz -C $PROJECT_HOME/data/

