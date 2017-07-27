#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
THOR detects differential peaks in multiple ChIP-seq profiles associated
with two distinct biological conditions.

Copyright (C) 2014-2016 Manuel Allhoff (allhoff@aices.rwth-aachen.de)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

@author: Manuel Allhoff
"""

from __future__ import print_function
from rgt.Util import npath
from rgt.Util import GenomeData
import errno
import os


def get_data_block(filepath, feature):
    with open(filepath) as f:
        data = []
        read = False
        for line in f:
            line = line.strip()
            if line.startswith("#") and line == "#" + str(feature):
                read = True
            
            if line.startswith("#") and line != "#" + str(feature):
                read = False
            
            if not line.startswith("#") and read:
                data.append(line)
                
    if len(data) == 1 and not (feature == "sample1" or feature == "sample2" or feature == "inputs1" or feature == "inputs2"):
        return data[0]
    else:
        return data


def input_parser(filepath):
    # read first replicate and get the absolute file path
    bamfiles_1 = get_data_block(filepath, "sample1")
    bamfiles_1 = map(npath, bamfiles_1)
    # replicate samples
    if len(bamfiles_1) == 1:
        bamfiles_1.append(bamfiles_1)

    bamfiles_2 = get_data_block(filepath, "sample2")
    bamfiles_2 = map(npath, bamfiles_2)
    if len(bamfiles_2) == 1:
        bamfiles_2.append(bamfiles_2)

    ## read chrom_sizes from config.file
   # chrom_sizes = npath(get_data_block(filepath, "chrom_sizes"))
   # chrom_sizes = npath(chrom_sizes) if chrom_sizes else chrom_sizes
    ## read chrom_sizes from #organism argument
    organism_name = get_data_block(filepath, "organism") # get organism name

    inputs1 = get_data_block(filepath, "inputs1")
    inputs1 = map(npath, inputs1)

    inputs2 = get_data_block(filepath, "inputs2")
    inputs2 = map(npath, inputs2)

    dims = [len(bamfiles_1), len(bamfiles_2)]
    
    if not inputs1 and not inputs2:
        inputs = None
    else:
        inputs = inputs1 + inputs2

    return bamfiles_1 + bamfiles_2, organism_name, inputs, dims



