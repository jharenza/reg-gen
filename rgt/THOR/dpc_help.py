"""
%prog [options] CONFIG

THOR detects differential peaks in multiple ChIP-seq profiles associated
with two distinct biological conditions.

Copyright (C) 2014-2016  Manuel Allhoff (allhoff@aices.rwth-aachen.de)

This program comes with ABSOLUTELY NO WARRANTY. This is free 
software, and you are welcome to redistribute it under certain 
conditions. Please see LICENSE file for details.
"""

# Python
from __future__ import print_function
import os
import sys
import pysam
import numpy as np
from math import fabs, log, ceil
from operator import add
from os.path import splitext, basename, join, isfile, isdir, exists
from optparse import OptionParser, OptionGroup
from datetime import datetime

# Internal
from rgt.THOR.postprocessing import merge_delete, filter_deadzones
from MultiCoverageSet import MultiCoverageSet
from rgt.GenomicRegionSet import GenomicRegionSet
from rgt.THOR.get_extension_size import get_extension_size
from rgt.THOR.get_fast_gen_pvalue import get_log_pvalue_new
from input_parser import input_parser
from rgt.Util import which, npath
from rgt import __version__

# External
from numpy import linspace
from scipy.optimize import curve_fit
import matplotlib as mpl
from rgt.Util import GenomeData
#see http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
mpl.use('Agg')
import matplotlib.pyplot as plt

FOLDER_REPORT = None


def merge_output(bamfiles, dims, options, no_bw_files, chrom_sizes):
    """it uses the number of input bamfiles to check output files """
    for i in range(len(bamfiles)):
        rep = i if i < dims[0] else i - dims[0]
        sig = 0 if i < dims[0] else 1

        temp_bed = npath(options.name + '_%s_rep%s_temp.bed' % (options.outputlabel[sig], rep+1))

        files = [options.name + '_' + str(j) + '_%s_rep%s.bw' %(options.outputlabel[sig], rep+1) for j in no_bw_files]
        if len(no_bw_files) > len(bamfiles):
            files = filter(lambda x: isfile(x), files)
            t = ['bigWigMerge'] + files + [temp_bed]
            c = " ".join(t)
            os.system(c)

            os.system("LC_COLLATE=C sort -k1,1 -k2,2n " + temp_bed + ' > ' + temp_bed +'.sort')

            t = ['bedGraphToBigWig', temp_bed + '.sort', chrom_sizes, options.name + '_%s_rep%s.bw' % (options.outputlabel[sig], rep+1)]
            c = " ".join(t)
            os.system(c)

            for f in files:
                os.remove(f)
            os.remove(temp_bed)
            os.remove(temp_bed + ".sort")
        else:
            ftarget = [options.name + '_%s_rep%s.bw' %(options.outputlabel[sig], rep+1) for j in no_bw_files]
            for i in range(len(ftarget)):
                c = ['mv', files[i], ftarget[i]]
                c = " ".join(c)
                os.system(c)


def _func_quad_2p(x, a, c):
    """Return y-value of y=max(|a|*x^2 + x + |c|, 0),
    x may be an array or a single float"""
    res = []
    if type(x) is np.ndarray:
        for el in x:
            res.append(max(el, fabs(a) * el**2 + el + fabs(c)))
            
        return np.asarray(res)
    else:
        return max(x, fabs(a) * x**2 + x + fabs(c))


def _write_emp_func_data(data, name):
    """Write mean and variance data"""
    assert len(data[0]) == len(data[1])
    f = open(FOLDER_REPORT_DATA + name + '.data', 'w')
    for i in range(len(data[0])):
        print(data[0][i], data[1][i], sep='\t', file=f)
    f.close()


def _plot_func(plot_data, outputdir):
    """Plot estimated and empirical function"""

    maxs = [] #max for x (mean), max for y (var)
    for i in range(2): 
        tmp = np.concatenate((plot_data[0][i], plot_data[1][i])) #plot_data [(m, v, p)], 2 elements
        maxs.append(max(tmp[tmp < np.percentile(tmp, 90)]))

    for i in range(2):
        x = linspace(0, max(plot_data[i][0]), int(ceil(max(plot_data[i][0]))))
        y = _func_quad_2p(x, plot_data[i][2][0], plot_data[i][2][1])
        
        for j in range(2):
            #use matplotlib to plot function and datapoints
            #and save datapoints to files
            ext = 'original'
            if j == 1:
                plt.xlim([0, maxs[0]])
                plt.ylim([0, maxs[1]])
                ext = 'norm'
            ax = plt.subplot(111)
            plt.plot(x, y, 'r', label = 'fitted polynomial') #plot polynom
            plt.scatter(plot_data[i][0], plot_data[i][1], label = 'empirical datapoints') #plot datapoints
            ax.legend()
            plt.xlabel('mean')
            plt.ylabel('variance')
            plt.title('Estimated Mean-Variance Function')
            name = "_".join(['mean', 'variance', 'func', 'cond', str(i), ext])
            _write_emp_func_data(plot_data[i], name)
            plt.savefig(FOLDER_REPORT_PICS + name + '.png')
            plt.close()


def _get_data_rep(overall_coverage, name, debug, sample_size):
    """Return list of (mean, var) points for samples 0 and 1"""
    data_rep = []
    for i in range(2):
        cov = np.asarray(overall_coverage[i]) #matrix: (#replicates X #bins)
        h = np.invert((cov==0).all(axis=0)) #assign True to columns != (0,..,0)
        cov = cov[:,h] #remove 0-columns

        r = np.random.randint(cov.shape[1], size=sample_size)
        r.sort()
        cov = cov[:,r]

        m = list(np.squeeze(np.asarray(np.mean(cov*1.0, axis=0))))
        n = list(np.squeeze(np.asarray(np.var(cov*1.0, axis=0))))
        assert len(m) == len(n)

        data_rep.append(zip(m, n))
        data_rep[i].append((0,0))
        data_rep[i] = np.asarray(data_rep[i])
        
    if debug:
        for i in range(2):
            np.save(str(name) + "-emp-data" + str(i) + ".npy", data_rep[i])
    
    for i in range(2):
        data_rep[i] = data_rep[i][data_rep[i][:,0] < np.percentile(data_rep[i][:,0], 99.75)]
        data_rep[i] = data_rep[i][data_rep[i][:,1] < np.percentile(data_rep[i][:,1], 99.75)]
    
    return data_rep


def _fit_mean_var_distr(overall_coverage, name, debug, verbose, outputdir, report, poisson, sample_size=5000):
    """Estimate empirical distribution (quadr.) based on empirical distribution"""
    done = False
    plot_data = [] #means, vars, paras
        
    while not done:
        data_rep = _get_data_rep(overall_coverage, name, debug, sample_size)
        res = []
        for i in range(2):
            try:
                m = np.asarray(map(lambda x: x[0], data_rep[i])) #means list
                v = np.asarray(map(lambda x: x[1], data_rep[i])) #vars list
                
                if len(m) > 0 and len(v) > 0: 
                    try:
                        p, _ = curve_fit(_func_quad_2p, m, v) #fit quad. function to empirical data
                    except:
                        print("Optimal parameters for mu-var-function not found, get new datapoints", file=sys.stderr)
                        break #restart for loop
                else:
                    p = np.array([0, 1])
                
                res.append(p)
                plot_data.append((m, v, p))
                if i == 1:
                    done = True
            except RuntimeError:
                print("Optimal parameters for mu-var-function not found, get new datapoints", file=sys.stderr)
                break #restart for loop
    
    if report:
        _plot_func(plot_data, outputdir)
    
    if poisson:
        print("Use Poisson distribution as emission", file=sys.stderr)
        p[0] = 0
        p[1] = 0
        res = [np.array([0, 0]), np.array([0, 0])]
    
    return lambda x: _func_quad_2p(x, p[0], p[1]), res

    
def dump_posteriors_and_viterbi(name, posteriors, DCS, states):
    print("Computing info...", file=sys.stderr)
    f = open(name + '-posts.bed', 'w')
    g = open(name + '-states-viterbi.bed', 'w')
    
    for i in range(len(DCS.indices_of_interest)):
        cov1, cov2 = _get_covs(DCS, i)
        p1, p2, p3 = posteriors[i][0], posteriors[i][1], posteriors[i][2]
        chrom, start, end = DCS._index2coordinates(DCS.indices_of_interest[i])
        
        print(chrom, start, end, states[i], cov1, cov2, sep='\t', file=g)
        print(chrom, start, end, max(p3, max(p1,p2)), p1, p2, p3, cov1, cov2, sep='\t', file=f)

    f.close()
    g.close()


def _compute_pvalue((x, y, side, distr)):
    a, b = int(np.mean(x)), int(np.mean(y))
    return -get_log_pvalue_new(a, b, side, distr)


def _get_log_ratio(l1, l2):
    l1, l2 = float(np.sum(np.array(l1))), float(np.sum(np.array(l2)))
    try:
        res = l1/l2
    except:
        return sys.maxint
    
    if res > 0:
        try:
            res = log(res)
            if np.isinf(res):
                return sys.maxint
            return res
        except:
            print('error to compute log ratio', l1, l2, file=sys.stderr)
            return sys.maxint
    else:
        return sys.maxint

def _merge_consecutive_bins(tmp_peaks, distr, merge=True):
    """Merge consecutive peaks and compute p-value. Return list
    <(chr, s, e, c1, c2, strand)> and <(pvalue)>"""
    nopcutoff_peaks = {}
    nopcutoff_pvalues = {}
    peaks = []
    pvalues = []
    i, j, = 0, 0

    while i < len(tmp_peaks):
        j+=1
        c, s, e, c1, c2, strand, strand_pos, strand_neg = tmp_peaks[i]
        v1 = c1
        v2 = c2

        tmp_pos = [strand_pos]
        tmp_neg = [strand_neg]
        #merge bins
        while merge and i+1 < len(tmp_peaks) and e == tmp_peaks[i+1][1] and strand == tmp_peaks[i+1][5]:
            e = tmp_peaks[i+1][2]
            v1 = map(add, v1, tmp_peaks[i+1][3])
            v2 = map(add, v2, tmp_peaks[i+1][4])
            tmp_pos.append(tmp_peaks[i+1][6])
            tmp_neg.append(tmp_peaks[i+1][7])
            i += 1

        side = 'l' if strand == '+' else 'r'

        if side == 'l':
            assert sum(v1) > sum(v2), "cov1 %s  cov2%s, %d" % (str(v1), str(v2), i)
        else:
            assert sum(v1) <= sum(v2), "cov1 %s  cov2%s, %d" % (str(v1), str(v2), i)

        pvalues.append((v1, v2, side, distr))

        ratio = _get_log_ratio(tmp_pos, tmp_neg)
        peaks.append((c, s, e, v1, v2, strand, ratio))
        i += 1

    pvalues = map(_compute_pvalue, pvalues)
    assert len(pvalues) == len(peaks)
    # we define all output format for the data
    nopcutoff_pvalues['all'] = pvalues
    nopcutoff_peaks['all'] = peaks
    return nopcutoff_pvalues, nopcutoff_peaks


def _calpvalues_merge_bins(tmp_peaks, bin_pvalues, distr, pcutoff):
    """ we calculate firstly p-values for each bin and then filter them using pcutoff values,
    at end we merge bins with similar p-values
    pcutoff: two format, one is just one value, secondly, one array, [start, end, steps]
    then above them, we get it, and then merge them..

    :tem_peaks
        tmp_peaks.append((chrom, start, end, cov1, cov2, strand, cov1_strand, cov2_strand))
        c, s, e, c1, c2, strand, strand_pos, strand_neg = tmp_peaks[i]
    :return
        ratio = _get_log_ratio(tmp_pos, tmp_neg)
        peaks.append((c, s, e, v1, v2, strand, ratio))
        pvalues = map(_compute_pvalue, pvalues)
    """

    pcutoff_peaks = {}
    pcutoff_pvalues = {}


    if len(pcutoff) == 3:
        pvalue_init, pvalue_end, pvalue_step = pcutoff[0], pcutoff[1],pcutoff[2]
        if pvalue_end > sys.maxint:
            pvalue_end = sys.maxint
    elif len(pcutoff) == 2:
        pvalue_init, pvalue_end, pvalue_step = pcutoff[0], pcutoff[1], 1
    elif len(pcutoff) == 1:
        pvalue_init, pvalue_end, pvalue_step = pcutoff[0], pcutoff[0] + 1, 1


    for i in np.arange(pvalue_init, pvalue_end, pvalue_step):
        new = 0
        new_peaks = []
        new_pvalues = []
        pi_peaks = []
        pi_pvalues = []
        for j in range(len(bin_pvalues)):
            # this means actually we filter like cumulatively..
            if bin_pvalues[j] >= i:
                new_pvalues.append(bin_pvalues[j])
                new_peaks.append(tmp_peaks[j])

        if new_pvalues is None:
            print('the setting of end filter p-value is too big')
            pcutoff_peaks['p_value_over_' + str(i)] = pi_peaks
            pcutoff_pvalues['p_value_over_' + str(i)] = pi_pvalues
            return pcutoff_peaks, pcutoff_pvalues

        for k in range(len(new_peaks)):
            chrom, s, e, ct1, ct2, strand, strand_pos, strand_neg = new_peaks[k]
            if new == 0:
                current_chr = chrom
                current_s = s
                current_e = e
                current_strand = strand
                current_ct1 = ct1
                current_ct2 = ct2
                current_strand_pos = [strand_pos]
                current_strand_neg = [strand_neg]
                current_pvalue = [new_pvalues[k]]
                new += 1
            else:
                if (current_chr == chrom) & (current_strand == strand) & (current_e >= s):
                    current_e = e
                    # if (numpy.argmax([pvalue,current_pvalue])==0):
                    current_pvalue.append(new_pvalues[k])
                    current_ct1 = np.add(ct1, current_ct1 )
                    current_ct2 = np.add(ct2, current_ct2 )
                    current_strand_pos += [strand_pos]
                    current_strand_neg += [strand_neg]
                    new += 1
                else:
                    # current_ct1=current_ct1/new
                    # current_ct2=current_ct2/new
                    current_pvalue.sort(reverse=True)
                    # print(current_pvalue)
                    p = - min([np.log10(new) - x - np.log10(j + 1) for j, x in enumerate(current_pvalue)])
                    ratio = _get_log_ratio(current_strand_pos, current_strand_neg)
                    pi_peaks.append((current_chr, current_s, current_e, list(current_ct1), list(current_ct2), current_strand, ratio))
                    pi_pvalues.append(p)

                    current_chr = chrom
                    current_s = s
                    current_e = e
                    current_strand = strand
                    current_ct1 = np.add(ct1, current_ct1)
                    current_ct2 = np.add(ct2, current_ct2)
                    current_strand_pos = [strand_pos]
                    current_strand_neg = [strand_neg]
                    current_pvalue = [new_pvalues[k]]

        current_pvalue.sort(reverse=True)
        p = - min([np.log10(new) - x - np.log10(j + 1) for j, x in enumerate(current_pvalue)])
        ratio = _get_log_ratio(current_strand_pos, current_strand_neg)
        pi_peaks.append((current_chr, current_s, current_e, list(current_ct1), list(current_ct2), current_strand, ratio))
        pi_pvalues.append(p)

        pcutoff_peaks['p_value_'+str(i)] = pi_peaks
        pcutoff_pvalues['p_value_'+str(i)] = pi_pvalues

    return  pcutoff_pvalues, pcutoff_peaks



def _get_covs(DCS, i, as_list=False):
    """For a multivariant Coverageset, return mean coverage cov1 and cov2 at position i"""
    if not as_list:
        cov1 = int(np.mean(DCS.overall_coverage[0][:, DCS.indices_of_interest[i]]))
        cov2 = int(np.mean(DCS.overall_coverage[1][:, DCS.indices_of_interest[i]]))
    else:
        cov1 = DCS.overall_coverage[0][:,DCS.indices_of_interest[i]]
        cov1 = map(lambda x: x[0], np.asarray((cov1)))
        cov2 = DCS.overall_coverage[1][:,DCS.indices_of_interest[i]]
        cov2 = map(lambda x: x[0], np.asarray((cov2)))
    
    return cov1, cov2


def get_peaks(name, DCS, states, exts, merge, distr, pcutoff, debug, no_correction, deadzones, merge_bin, p=70):
    """Merge Peaks, compute p-value and give out *.bed and *.narrowPeak
    pcutoff is used to ??
    p : filter the peaks to get the most significant peaks..
    """
    exts = np.mean(exts)
    tmp_peaks = []
    tmp_data = []
    
    for i in range(len(DCS.indices_of_interest)):
        if states[i] not in [1,2]:
            continue #ignore background states


        # here strand is used to represent the gain or lose peak...
        # but we don't want it to mess with BED file format, so what could we get this information, and give them then??
        # Firstly not just to split it into two files according to strand values..

        ## does state == 1 imply that cov1, cov2 has a relation  cov1 < cvo2 ??
        strand = '+' if states[i] == 1 else '-'
        cov1, cov2 = _get_covs(DCS, i, as_list=True)


        cov1_strand = np.sum(DCS.overall_coverage_strand[0][0][:,DCS.indices_of_interest[i]]) + np.sum(DCS.overall_coverage_strand[1][0][:,DCS.indices_of_interest[i]])
        cov2_strand = np.sum(DCS.overall_coverage_strand[0][1][:,DCS.indices_of_interest[i]] + DCS.overall_coverage_strand[1][1][:,DCS.indices_of_interest[i]])
        
        chrom, start, end = DCS._index2coordinates(DCS.indices_of_interest[i])


        tmp_peaks.append((chrom, start, end, cov1, cov2, strand, cov1_strand, cov2_strand))
        side = 'l' if strand == '+' else 'r'  ### actually the use of side is also not clear..
        tmp_data.append((sum(cov1), sum(cov2), side, distr))
    
    if not tmp_data:
        print('no data', file=sys.stderr)
        return [], [], []

    # here to calculate p-value for each bin, then get the 70% neg_log(p-values), which means smaller p-values,
    # after that, we combine them to new data..

    tmp_pvalues = map(_compute_pvalue, tmp_data)
    per = np.percentile(tmp_pvalues, p)
    
    tmp = []
    bin_pvalues = []
    res = tmp_pvalues > per
    for j in range(len(res)):
        if res[j]:
            tmp.append(tmp_peaks[j])
            bin_pvalues.append(tmp_pvalues[j])
    tmp_peaks = tmp


    # From here we give different choices, one is to use the old one A, one is for the new method B with pcutoff
    if pcutoff is None:
        print('use old method ( firstly merge bins and then calculate p-values for each merged bins )to calculate p-values')
        pvalues, peaks = _merge_consecutive_bins(tmp_peaks, distr, merge_bin) #merge consecutive peaks by coverage number integer, and compute p-value
    else:
        # we use pcutoff values to new method B, we get p-values already from this part, better to pass directly
        pvalues, peaks = _calpvalues_merge_bins(tmp_peaks, bin_pvalues, distr, pcutoff)


    # since using dictionary, the best way is to make old into old fashion
    # we define another dictionary item for it..'all' means nopcutoff..

    # this is code whic is used in merge_delete
    # chrom, start, end, c1, c2, strand, ratio = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
    #   r = GenomicRegion(chrom = chrom, initial = start, final = end, name = '', \
    #                     orientation = strand, data = str((c1, c2, pvalue_list[i], ratio)))
    # after this, we also need to define different output,
    peak_output = {}
    peak_pvalues = {}
    peak_ratios = {}

    for pi_value in pvalues.keys():

        regions = merge_delete(exts, merge, peaks[pi_value], pvalues[pi_value]) #postprocessing, returns GenomicRegionSet with merged regions
        if deadzones:
            regions = filter_deadzones(deadzones, regions)

        peak_output[pi_value] = []
        peak_pvalues[pi_value] = []
        peak_ratios[pi_value] = []
        main_sep = ':' #sep <counts> main_sep <counts> main_sep <pvalue>
        int_sep = ';' #sep counts in <counts>


        for i, el in enumerate(regions):
            tmp = el.data.split(',')
            counts = ",".join(tmp[0:len(tmp)-1]).replace('], [', int_sep).replace('], ', int_sep).replace('([', '').replace(')', '').replace(', ', main_sep)
            pvalue = float(tmp[len(tmp)-2].replace(")", "").strip())
            ratio = float(tmp[len(tmp)-1].replace(")", "").strip())
            peak_pvalues[pi_value].append(pvalue)
            peak_ratios[pi_value].append(ratio)
            peak_output[pi_value].append((el.chrom, el.initial, el.final, el.orientation, counts))

    return peak_ratios, peak_pvalues, peak_output

def _output_ext_data(ext_data_list, bamfiles):
    """Output textfile and png file of read size estimation"""
    names = [splitext(basename(bamfile))[0] for bamfile in bamfiles]

    for k, ext_data in enumerate(ext_data_list):
        f = open(FOLDER_REPORT_DATA + 'fragment_size_estimate_' + names[k] + '.data', 'w')
        for d in ext_data:
            print(d[0], d[1], sep='\t', file=f)
        f.close()
    
    for i, ext_data in enumerate(ext_data_list):
        d1 = map(lambda x: x[0], ext_data)
        d2 = map(lambda x: x[1], ext_data)
        ax = plt.subplot(111)
        plt.xlabel('shift')
        plt.ylabel('convolution')
        plt.title('Fragment Size Estimation')
        plt.plot(d2, d1, label=names[i])
    
    ax.legend()
    plt.savefig(FOLDER_REPORT_PICS + 'fragment_size_estimate.png')
    plt.close()


def _compute_extension_sizes(bamfiles, exts, inputs, exts_inputs, report):
    """Compute Extension sizes for bamfiles and input files
    @:param bamfiles:  list of banfiles for reads
    @:param exts: user defined Read's extension size list for BAM files
    @:param inputs: genome input files
    @:param exts_inputs : read extension size for input genome, !!
            but always empty list, cause there is no such parameter accepted from user
    @:param report:  if report is True, we write it html files
    :return
        exts: extension size list for reads which means fragment size of each bamfiles
        exts_inputs: input genomes are not empty, set ext_inputs to be [5,5,5,5.....] of length len(inputs)
    """
    start = 0
    end = 600
    ext_stepsize = 5

    ext_data_list = []
    #compute extension size
    if not exts:
        print("Computing read extension sizes for ChIP-seq profiles", file=sys.stderr)
        for bamfile in bamfiles:
            e, ext_data = get_extension_size(bamfile, start=start, end=end, stepsize=ext_stepsize)
            exts.append(e)
            ext_data_list.append(ext_data)
    
    if report and ext_data_list:
        _output_ext_data(ext_data_list, bamfiles)
    
    if inputs and not exts_inputs:
        exts_inputs = [5] * len(inputs)

    return exts, exts_inputs


def get_all_chrom(bamfiles):
    chrom = set()
    for bamfile in bamfiles:
        bam = pysam.Samfile(bamfile, "rb" )
        for read in bam.fetch():
            c = bam.getrname(read.reference_id)
            if c not in chrom:
                chrom.add(c)
    return chrom


def initialize(name, dims, genome_path, regions, stepsize, binsize, bamfiles, exts, \
               inputs, exts_inputs, factors_inputs, chrom_sizes, verbose, gc_correct, \
               tracker, debug, norm_regions, scaling_factors_ip, save_wig, housekeeping_genes, \
               test, report, chrom_sizes_dict, counter, end, gc_content_cov=None, avg_gc_content=None, \
               gc_hist=None, output_bw=True, save_input=False, m_threshold=80, a_threshold=95, rmdup=False):
    """Initialize the MultiCoverageSet which is after normalization like gc-content, subtraction by input genes"""
    regionset = regions
    regionset.sequences.sort()
    
    if norm_regions:
        norm_regionset = GenomicRegionSet('norm_regions')
        norm_regionset.read(norm_regions)
    else:
        norm_regionset = None
        
    exts, exts_inputs = _compute_extension_sizes(bamfiles, exts, inputs, exts_inputs, report)
    
    multi_cov_set = MultiCoverageSet(name=name, regions=regionset, dims=dims, genome_path=genome_path,
                                     binsize=binsize, stepsize=stepsize, rmdup=rmdup, path_bamfiles=bamfiles,
                                     path_inputs=inputs, exts=exts, exts_inputs=exts_inputs,
                                     factors_inputs=factors_inputs, chrom_sizes=chrom_sizes, verbose=verbose,
                                     gc_correct=gc_correct, chrom_sizes_dict=chrom_sizes_dict, debug=debug,
                                     norm_regionset=norm_regionset, scaling_factors_ip=scaling_factors_ip,
                                     save_wig=save_wig, strand_cov=True, housekeeping_genes=housekeeping_genes,
                                     tracker=tracker, gc_content_cov=gc_content_cov, avg_gc_content=avg_gc_content,
                                     gc_hist=gc_hist, end=end, counter=counter, output_bw=output_bw,
                                     folder_report=FOLDER_REPORT, report=report, save_input=save_input,
                                     m_threshold=m_threshold, a_threshold=a_threshold)
    return multi_cov_set


class HelpfulOptionParser(OptionParser):
    """An OptionParser that prints full help on errors."""
    def error(self, msg):
        self.print_help(sys.stderr)
        self.exit(2, "\n%s: error: %s\n" % (self.get_prog_name(), msg))

def _callback_list(option, opt, value, parser):
    setattr(parser.values, option.dest, map(lambda x: int(x), value.split(',')))

def _callback_list_string(option, opt, value, parser):
    assert value
    filenames = []
    filenames.append(value)

    for arg in parser.rargs:
        if arg[:2] == '--' and len(arg) > 2:
            break
        filenames.append(arg)

    setattr(parser.values, option.dest, filenames)

def _callback_list_float(option, opt, value, parser):
    setattr(parser.values, option.dest, map(lambda x: float(x), value.split(',')))


def handle_input():
    parser = HelpfulOptionParser(usage=__doc__)

    ## add bam option for inpout files : Required
    ## define callback_function to accept variable parameters
    parser.add_option("--bam1", dest="bamfiles1", default=None, type='str',  action='callback', callback=_callback_list_string,
                      help="Give input .bam files . [default: %default]")

    parser.add_option("--bam2", dest="bamfiles2", default=None, type='str',  action='callback', callback=_callback_list_string,
                      help="Give input .bam files . [default: %default]")

    #parser.add_option("--bam",  dest="bamfiles", default=None, type="str",  nargs=2,\
    #                  help="Give input .bam files of two samples, each replicates of every sample should be separeted by comma. [default: %default]")
    ## add chrom sizes : Required
    parser.add_option("--organism", dest="organism", type="str",
                      help="Give organism and we define chrom_sizes by it. [default: %default]")

    ## add label option to name output directory and files
    parser.add_option("-l", "--label", dest="outputlabel", default=["sample1", "sample2"], type="str", nargs=2,
                      help="Give labels to name output directory and files. [default: %default]")

    ## add genome to specify Input-DNA and the genome (in fasta format) is necessary to correct for GC-content.
    parser.add_option("--genome", dest="genome", default=None, type="str",
                      help="Give genome. [default: %default]")

    ## add inputs Input-DNA helps to handle bias in ChIP-seq profiles and can therefore improve the differential peak estimation.
    parser.add_option("--iDNA", dest="inputDNA", default=None, type="str",
                      help="Give inputDNA files to handle biases. [default: %default]")


    # Actually it does the similar work like labels
    parser.add_option("-n", "--name", default=None, dest="name", type="string",
                      help="Experiment's name and prefix for all files that are created.")
    parser.add_option("-m", "--merge", default=False, dest="merge", action="store_true",
                      help="Merge peaks which have a distance less than the estimated mean fragment size "
                           "(recommended for histone data). [default: do not merge]")
    parser.add_option("--no-merge-bin", default=True, dest="merge_bin", action="store_false",
                      help="Merge the overlapping bin before filtering by p-value."
                           "[default: Merging bins]")
    parser.add_option("--housekeeping-genes", default=None, dest="housekeeping_genes", type="str",
                      help="Define housekeeping genes (BED format) used for normalizing. [default: %default]")
    parser.add_option("--output-dir", dest="outputdir", default=None, type="string",
                      help="Store files in output directory. [default: %default]")
    parser.add_option("--report", dest="report", default=False, action="store_true",
                      help="Generate HTML report about experiment. [default: %default]")
    parser.add_option("--deadzones", dest="deadzones", default=None,
                      help="Define blacklisted genomic regions avoided for analysis (BED format). [default: %default]")
    parser.add_option("--no-correction", default=False, dest="no_correction", action="store_true",
                      help="Do not use multipe test correction for p-values (Benjamini/Hochberg). [default: %default]")

    parser.add_option("-p", "--pvalue", dest="pcutoff", default=None, type="str", action='callback',
                      callback=_callback_list_float,
                      help="P-value cutoff for peak detection. Call only peaks with p-value lower than cutoff. "
                           "[default: %default]")

    ## add p_fdr_correction option to substitute the original pcutoff
    parser.add_option("--p_fdr", "--p_fdr_correction", dest="p_fdr_correction", default=0.1, type="float",
                      help="fdr_correction for peak filtering and reduce false discovery rate "
                           "[default: %default]")

    parser.add_option("--exts", default=None, dest="exts", type="str", action='callback', callback=_callback_list,
                      help="Read's extension size for BAM files (comma separated list for each BAM file in config "
                           "file). If option is not chosen, estimate extension sizes. [default: %default]")
    parser.add_option("--factors-inputs", default=None, dest="factors_inputs", type="str", action="callback",
                      callback=_callback_list_float,
                      help="Normalization factors for input-DNA (comma separated list for each BAM file in config "
                           "file). If option is not chosen, estimate factors. [default: %default]")
    parser.add_option("--scaling-factors", default=None, dest="scaling_factors_ip", type="str", action='callback',
                      callback=_callback_list_float,
                      help="Scaling factor for each BAM file (not control input-DNA) as comma separated list for "
                           "each BAM file in config file. If option is not chosen, follow normalization strategy "
                           "(TMM or HK approach) [default: %default]")
    parser.add_option("--save-input", dest="save_input", default=False, action="store_true",
                      help="Save input-DNA file if available. [default: %default]")
    parser.add_option("--version", dest="version", default=False, action="store_true",
                      help="Show script's version.")

    group = OptionGroup(parser, "Advanced options")
    group.add_option("--regions", dest="regions", default=None, type="string",
                     help="Define regions (BED format) to restrict the analysis, that is, where to train the HMM and "
                          "search for DPs. It is faster, but less precise.")
    group.add_option("-b", "--binsize", dest="binsize", default=100, type="int",
                     help="Size of underlying bins for creating the signal. [default: %default]")
    group.add_option("-s", "--step", dest="stepsize", default=50, type="int",
                     help="Stepsize with which the window consecutively slides across the genome to create the "
                          "signal. [default: %default]")
    group.add_option("--debug", default=False, dest="debug", action="store_true",
                     help="Output debug information. Warning: space consuming! [default: %default]")
    group.add_option("--gc_correct", dest="gc_correct", default=False, action="store_true",
                     help="Normalize towards GC content. [default: %default]")
    group.add_option("--norm-regions", default=None, dest="norm_regions", type="str",
                     help="Restrict normalization to particular regions (BED format). [default: %default]")
    group.add_option("-f", "--foldchange", dest="foldchange", default=1.6, type="float",
                     help="Fold change parameter to define training set (t_1, see paper). [default: %default]")
    group.add_option("-t", "--threshold", dest="threshold", default=95, type="float",
                     help="Minimum signal support for differential peaks to define training set as percentage "
                          "(t_2, see paper). [default: %default]")
    group.add_option("--size", dest="size_ts", default=10000, type="int",
                     help="Number of bins the HMM's training set constists of. [default: %default]")
    group.add_option("--par", dest="par", default=1, type="int",
                     help="Percentile for p-value postprocessing filter. [default: %default]")
    group.add_option("--poisson", default=False, dest="poisson", action="store_true",
                     help="Use binomial distribution as emmission. [default: %default]")
    group.add_option("--single-strand", default=False, dest="singlestrand", action="store_true",
                     help="Allow single strand BAM file as input. [default: %default]")
    group.add_option("--m_threshold", default=80, dest="m_threshold", type="int",
                     help="Define the M threshold of percentile for training TMM. [default: %default]")
    group.add_option("--a_threshold", default=95, dest="a_threshold", type="int",
                     help="Define the A threshold of percentile for training TMM. [default: %default]")
    group.add_option("--rmdup", default=False, dest="rmdup", action="store_true",
                     help="Remove the duplicate reads [default: %default]")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()
    options.save_wig = False
    options.exts_inputs = None
    options.verbose = False
    options.hmm_free_para = False

    if options.version:
        print("")
        print(__version__)
        sys.exit()


    # config file is required and then we read config file and get parameters
    if len(args) == 1:
        # parser.error("Please give config file")
        config_path = npath(args[0])
        if isfile(config_path):
            # parser.error("Config file %s does not exist!" % config_path)
                 bamfiles, organism_name, inputs, dims = input_parser(config_path)
        else:
            parser.error("Config file %s does not exist!" % config_path)
    else:
        # Now we want to change to command line methods..and I would like to define another function
        # with function there is a problem that we need to pass a lot of parameters. then at first write here to achieve this
        if not options.bamfiles1 or not options.bamfiles2:
            parser.error('BamFiles not given')
        else:
            # this code is used to replicate the samples if input sample just one  under each condition
            if len(options.bamfiles1) == 1:
                options.bamfiles1.append(options.bamfiles1)
            if len(options.bamfiles2) == 1:
                options.bamfiles2.append(options.bamfiles2)

            bamfiles = options.bamfiles1 + options.bamfiles2
            dims = [len(options.bamfiles1), len(options.bamfiles2)]

        if not options.organism:
            parser.error('Organism not given')
        organism_name = options.organism

        # set inputs parameters
        inputs = None
        if options.inputDNA:
            inputs_tmp = map(lambda x: x.split(','), options.inputDNA)
            inputs = map(npath, inputs_tmp[0] + inputs_tmp[1])

    # Now we want to change to command line methods..and I would like to define another function
    # with function there is a problem that we need to pass a lot of parameters. then at first write here to achieve this

    organism = GenomeData(organism=organism_name)

    if organism is None:
        parser.error("organism doesn't exist, please install it firstly")

    chrom_sizes = organism.get_chromosome_sizes()
    ## If we want to do gc-correct, then we need input genome named inputs!!!
    if options.gc_correct:
        if inputs is None:
            parser.error(" Doing gc_correct needs inputs genome file")
        genome = organism.get_genome()
    else:
        genome = None


    if options.exts and len(options.exts) != len(bamfiles):
        parser.error("Number of Extension Sizes must equal number of bamfiles")

    if options.exts_inputs and len(options.exts_inputs) != len(inputs):
        parser.error("Number of Input Extension Sizes must equal number of input bamfiles")

    if options.scaling_factors_ip and len(options.scaling_factors_ip) != len(bamfiles):
        parser.error("Number of scaling factors for IP must equal number of bamfiles")

    # check if it is right
   # print('check the output of bamfiles')
   # print(bamfiles)
    for bamfile in bamfiles:
        if not isfile(bamfile):
            parser.error(" BAM file %s does not exist!" % bamfile)

    if not inputs and options.factors_inputs:
        print("As no input-DNA, do not use input-DNA factors", file=sys.stderr)
        options.factors_inputs = None

    if options.factors_inputs and len(options.factors_inputs) != len(bamfiles):
        parser.error("factors for input-DNA must equal number of BAM files!")

    if inputs:
        for bamfile in inputs:
            if not isfile(bamfile):
                parser.error("BAM Inputs file %s does not exist!" % bamfile)

    if options.regions:
        if not isfile(options.regions):
            parser.error("Region file %s does not exist!" % options.regions)

    # This code defines the name for file name using time stamp.
    #
#    if options.name is None:
#       d = str(datetime.now()).replace("-", "_").replace(":", "_").replace(" ", "_").replace(".", "_").split("_")
#        options.name = "THOR-exp" + "-" + "_".join(d[:len(d) - 1])

    #  But we want to change it, which is semantic
    # use labels to name files
    if options.name is None:
        # print(options.outputlabel)
        options.name ='_vs_'.join(options.outputlabel)

    if not which("wigToBigWig") or not which("bedGraphToBigWig") or not which("bigWigMerge"):
        print("Warning: wigToBigWig, bigWigMerge or bedGraphToBigWig not found! Signal will not be stored!",
              file=sys.stderr)

    if options.outputdir:
        options.outputdir = npath(options.outputdir)
        if isdir(options.outputdir) and sum(
                map(lambda x: x.startswith(options.name), os.listdir(options.outputdir))) > 0:
            parser.error("Output directory exists and contains files with names starting with your chosen experiment "
                         "name! Do nothing to prevent file overwriting!")
    else:
        ## here we create the folder by default using the labels...Not necessary
        options.outputdir = os.getcwd() +'/'+ options.name + '/'

    if not exists(options.outputdir):
        os.mkdir(options.outputdir)

    # print(options.outputdir)

    options.name = join(options.outputdir, options.name)

    if options.report and isdir(join(options.outputdir, 'report_'+basename(options.name))):
        parser.error("Folder 'report_"+basename(options.name)+"' already exits in output directory!" 
                     "Do nothing to prevent file overwriting! "
                     "Please rename report folder or change working directory of THOR with the option --output-dir")

    if options.report:
        os.mkdir(join(options.outputdir, 'report_'+basename(options.name)+"/"))
        os.mkdir(join(options.outputdir, 'report_'+basename(options.name), 'pics/'))
        os.mkdir(join(options.outputdir, 'report_'+basename(options.name), 'pics/data/'))


    global FOLDER_REPORT
    global FOLDER_REPORT_PICS
    global FOLDER_REPORT_DATA
    global OUTPUTDIR
    global NAME

    FOLDER_REPORT = join(options.outputdir, 'report_'+basename(options.name)+"/")
    FOLDER_REPORT_PICS = join(options.outputdir, 'report_'+basename(options.name), 'pics/')
    FOLDER_REPORT_DATA = join(options.outputdir, 'report_'+basename(options.name), 'pics/data/')
    OUTPUTDIR = options.outputdir
    NAME = options.name

    if not inputs:
        print("Warning: Do not compute GC-content, as there is no input file", file=sys.stderr)

    if not genome:
        print("Warning: Do not compute GC-content, cause gc-correct is False", file=sys.stderr)

    if options.exts is None:
        options.exts = []

    if options.exts_inputs is None:
        options.exts_inputs = []

    return options, bamfiles, genome, organism_name, chrom_sizes, dims, inputs


if __name__ == '__main__':
    handle_input()
