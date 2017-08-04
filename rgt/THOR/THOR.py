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

# Python
from __future__ import print_function
import sys

# Internal
from dpc_help import get_peaks, _fit_mean_var_distr, initialize, merge_output, handle_input
from tracker import Tracker
from postprocessing import _output_BED, _output_narrowPeak, _output_empty_BED
from rgt.THOR.neg_bin_rep_hmm import NegBinRepHMM, get_init_parameters, _get_pvalue_distr
from rgt.THOR.RegionGiver import RegionGiver
from rgt.THOR.postprocessing import filter_by_pvalue_strand_lag
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.GenomicRegion import GenomicRegion

from rgt import __version__

# External


TEST = False #enable to test THOR locally


def _write_info(tracker, report, **data):
    """Write information to tracker"""
    tracker.write(text=data['func_para'][0], header="Parameters for both estimated quadr. function y=max(|a|*x^2 + x + |c|, 0) (a)")
    tracker.write(text=data['func_para'][1], header="Parameters for both estimated quadr. function y=max(|a|*x^2 + x + |c|, 0) (c)")
    #tracker.write(text=data['init_mu'], header="Inital parameter estimate for HMM's Neg. Bin. Emission distribution (mu)")
    #tracker.write(text=data['init_alpha'], header="Inital parameter estimate for HMM's Neg. Bin. Emission distribution (alpha)")
    #tracker.write(text=data['m'].mu, header="Final HMM's Neg. Bin. Emission distribution (mu)")
    #tracker.write(text=data['m'].alpha, header="Final HMM's Neg. Bin. Emission distribution (alpha)")
    #tracker.write(text=data['m']._get_transmat(), header="Transmission matrix")
    
    if report:
        tracker.make_html()


def train_HMM(region_giver, options, bamfiles, genome, chrom_sizes, dims, inputs, tracker):
    """Train HMM and get most possible state sequence from given data using Viterbi algorithm
    :return m:
    :return exp_data:
    :return func_para
    :return init_mu
    :return inti_alpha
    :return distr
    """
    
    while True:
        train_regions = region_giver.get_training_regionset()
        exp_data = initialize(name=options.name, dims=dims, genome_path=genome, regions=train_regions,
                              stepsize=options.stepsize, binsize=options.binsize, bamfiles=bamfiles,
                              exts=options.exts, inputs=inputs, exts_inputs=options.exts_inputs,
                              debug=options.debug, verbose=options.verbose, gc_correct=options.gc_correct,
                              factors_inputs=options.factors_inputs, chrom_sizes=chrom_sizes,
                              tracker=tracker, norm_regions=options.norm_regions,
                              scaling_factors_ip=options.scaling_factors_ip, save_wig=options.save_wig,
                              housekeeping_genes=options.housekeeping_genes, test=TEST, report=options.report,
                              chrom_sizes_dict=region_giver.get_chrom_dict(), end=True, counter=0, output_bw=False,
                              save_input=options.save_input, m_threshold=options.m_threshold,
                              a_threshold=options.a_threshold, rmdup=options.rmdup)
        ## After initialization, we get normalized training data (matrix)
        if exp_data.count_positive_signal() > len(train_regions.sequences[0]) * 0.00001:
            tracker.write(text=" ".join(map(lambda x: str(x), exp_data.exts)), header="Extension size (rep1, rep2, input1, input2)")
            tracker.write(text=map(lambda x: str(x), exp_data.scaling_factors_ip), header="Scaling factors")
            break
    
    func, func_para = _fit_mean_var_distr(exp_data.overall_coverage, options.name, options.debug,
                                          verbose=options.verbose, outputdir=options.outputdir,
                                          report=options.report, poisson=options.poisson)
    exp_data.compute_putative_region_index()
     
    print('Compute HMM\'s training set', file=sys.stderr)
    training_set, s0, s1, s2 = exp_data.get_training_set(TEST, exp_data, options.name, options.foldchange,
                                                         options.threshold, options.size_ts, 3)
    init_alpha, init_mu = get_init_parameters(s0, s1, s2)
    m = NegBinRepHMM(alpha=init_alpha, mu=init_mu, dim_cond_1=dims[0], dim_cond_2=dims[1], func=func)
    training_set_obs = exp_data.get_observation(training_set)
     
    print('Train HMM', file=sys.stderr)
    m.fit([training_set_obs], options.hmm_free_para)
    distr = _get_pvalue_distr(m.mu, m.alpha, tracker)
         
    return m, exp_data, func_para, init_mu, init_alpha, distr


def run_HMM(region_giver, options, bamfiles, genome, organism_name, chrom_sizes, dims, inputs, tracker, exp_data, m, distr):
    """Run trained HMM chromosome-wise on genomic signal and call differential peaks"""
    # output, pvalues, ratios, no_bw_files = [], [], [], []
    print("Compute HMM's posterior probabilities and Viterbi path to call differential peaks", file=sys.stderr)
    pcutoff_output, pcutoff_pvalues, pcutoff_ratios = {}, {}, {}
    no_bw_files = []

    for i, r in enumerate(region_giver):
        end = True if i == len(region_giver) - 1 else False
        print("- taking into account %s" % r.sequences[0].chrom, file=sys.stderr)
        
        exp_data = initialize(name=options.name, dims=dims, genome_path=genome, regions=r,
                              stepsize=options.stepsize, binsize=options.binsize,
                              bamfiles=bamfiles, exts=exp_data.exts, inputs=inputs,
                              exts_inputs=exp_data.exts_inputs, debug=options.debug,
                              verbose=False, gc_correct=options.gc_correct,
                              factors_inputs=exp_data.factors_inputs, chrom_sizes=chrom_sizes,
                              tracker=tracker, norm_regions=options.norm_regions,
                              scaling_factors_ip=exp_data.scaling_factors_ip, save_wig=options.save_wig,
                              housekeeping_genes=options.housekeeping_genes, test=TEST, report=False,
                              chrom_sizes_dict=region_giver.get_chrom_dict(), gc_content_cov=exp_data.gc_content_cov,
                              avg_gc_content=exp_data.avg_gc_content, gc_hist=exp_data.gc_hist,
                              end=end, counter=i, m_threshold=options.m_threshold, a_threshold=options.a_threshold,
                              rmdup=options.rmdup)
        if exp_data.no_data:
            continue
        no_bw_files.append(i)
        exp_data.compute_putative_region_index()

        if exp_data.indices_of_interest is None:
            continue
        
        states = m.predict(exp_data.get_observation(exp_data.indices_of_interest))
        
        inst_ratios, inst_pvalues, inst_output = get_peaks(name=options.name, states=states, DCS=exp_data,
                                                           distr=distr, merge=options.merge, exts=exp_data.exts,
                                                           pcutoff=options.pcutoff, debug=options.debug, p=options.par,
                                                           no_correction=options.no_correction,
                                                           merge_bin=options.merge_bin, deadzones=options.deadzones)

        # 21 July 2017
        # cause I change output as dictionary, so this should change to dictionary addition
        # output just for old method A is to do:
        for pi_value in inst_output.keys():
            if pi_value not in pcutoff_output.keys():
                pcutoff_output[pi_value] = []
                pcutoff_pvalues[pi_value] = []
                pcutoff_ratios[pi_value] = []

            pcutoff_output[pi_value] += inst_output[pi_value]
            pcutoff_pvalues[pi_value] += inst_pvalues[pi_value]
            pcutoff_ratios[pi_value] += inst_ratios[pi_value]
        """
        output += inst_output
        pvalues += inst_pvalues
        ratios += inst_ratios
        """
        # read_bed: z.add(GenomicRegion(chrom, start, end, name, orientation, data))
        # getpeak:  output.append((el.chrom, el.initial, el.final, el.orientation, counts))

    ## Here is the place to create the output files which is
    ## output.append((el.chrom, el.initial, el.final, el.orientation, counts)), so the names of it just the this info..
    ## we could change it here,
    # firstly to generate GenomicSet again and use the association function and add the names already there, and done
    ## how could we get it organism ?? Integrated??

    ## 21 July 2017
    # need to deal with dictionary data..one by one is the best way..
    res_output, res_pvalues, res_filter_pass = {}, {}, {}
    for pi_value in pcutoff_output.keys():
        ## add confition if pi_value is empty, we give empty output directly
        if pcutoff_output[pi_value] is None:
            _output_empty_BED(options.name + '_' + pi_value)
            break

        peakregset = GenomicRegionSet(pi_value+"_peaks")
        for item in pcutoff_output[pi_value]:
            # print(item)
            peakregset.add(
                GenomicRegion(item[0], item[1], int(item[2]), '', item[3], item[4]))

        pcutoff_output[pi_value] = peakregset.gene_association(organism=organism_name, promoterLength=1000,
                                                 threshDist=500000, show_dis=True)
        res_output[pi_value], res_pvalues[pi_value], res_filter_pass[pi_value] = filter_by_pvalue_strand_lag(pcutoff_ratios[pi_value], options.p_fdr_correction, pcutoff_pvalues[pi_value],
                                                                               pcutoff_output[pi_value],
                                                                               options.no_correction, options.name +'_'+pi_value,
                                                                               options.singlestrand)

        # print('progress before output_BED files')
        _output_BED(options.name+'_'+pi_value, res_output[pi_value], res_pvalues[pi_value], res_filter_pass[pi_value])
        # print('progress before narrow output_BED files')
        _output_narrowPeak(options.name+'_'+pi_value, res_output[pi_value], res_pvalues[pi_value], res_filter_pass[pi_value])

    # res_output, res_pvalues, res_filter_pass = filter_by_pvalue_strand_lag(ratios, options.p_fdr_corretion, pvalues, output,
    #                                                                       options.no_correction, options.name,
    #                                                                      options.singlestrand)

    merge_output(bamfiles, dims, options, no_bw_files, chrom_sizes)


def main():
    options, bamfiles, genome, organism_name, chrom_sizes, dims, inputs = handle_input()

    tracker = Tracker(options.name + '_setup.info', bamfiles, genome, chrom_sizes, dims, inputs, options, __version__)
    region_giver = RegionGiver(chrom_sizes, options.regions)
    m, exp_data, func_para, init_mu, init_alpha, distr = train_HMM(region_giver, options, bamfiles, genome,
                                                                   chrom_sizes, dims, inputs, tracker)
    
    run_HMM(region_giver, options, bamfiles, genome, organism_name,  chrom_sizes, dims, inputs, tracker, exp_data, m, distr)

    # bed2associated_genes(options.outputdir, organism_name), it is used to deal with the output files
    _write_info(tracker, options.report, func_para=func_para, init_mu=init_mu, init_alpha=init_alpha, m=m)

if __name__ == "__main__":
    main()
