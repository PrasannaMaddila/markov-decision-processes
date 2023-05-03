#!/bin/bash
# intialisation script to create the environment and 
# install dependencies as found in requirements.txt
# Author : Prasanna <maddilaprasanna10@gmail.com>
# Created: 03 May 2023

virtualenv .env && source .env/bin/activate && pip install -r requirements.txt
