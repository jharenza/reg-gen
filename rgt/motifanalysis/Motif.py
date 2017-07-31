
###################################################################################################
# Libraries
###################################################################################################

# Python
from os.path import basename

# Internal
from rgt.Util import ErrorHandler

# External
from Bio import motifs
import MOODS.tools
import MOODS.parsers

###################################################################################################
# Classes
###################################################################################################


class Motif:
    """
    Represent a DNA binding affinity motif.
    """

    def __init__(self, input_file_name, pseudocounts, precision, fpr, thresholds):
        """ 
        Initializes Motif.

        Variables:
        pfm -- Position Frequency Matrix.
        pwm -- Position Weight Matrix.
        pssm -- Position Specific Scoring Matrix.
        alphabet -- A list of letters, eg ["A", "C", "G", "T"]
        threshold -- Motif matching threshold.
        len -- Length of the motif.
        max -- Maximum PSSM score possible.
        is_palindrome -- True if consensus is biologically palindromic.
        """

        # Initializing error handler
        err = ErrorHandler()
 
        # Initializing name
        self.name = ".".join(basename(input_file_name).split(".")[:-1])
        repository = input_file_name.split("/")[-2]

        # Creating PFM & PSSM
        self.pfm = MOODS.parsers.pfm(input_file_name)
        self.bg = MOODS.tools.flat_bg(len(self.pfm))  # total number of "points" to add, not per-row
        self.pssm = MOODS.tools.log_odds(self.pfm, self.bg, pseudocounts)
        self.max = max([max(e) for e in self.pssm])

        # Evaluating threshold
        # TODO: must probably recalculate all thresholds using MOODS functions (there's a script somewhere)
        try:
            if pseudocounts != 0.1 or precision != 10000:
                raise ValueError()
            self.threshold = thresholds.dict[repository][self.name][fpr]
        except Exception:
            err.throw_warning("DEFAULT_WARNING", add_msg="Parameters not matching pre-computed Fpr data. "
                                                         "Recalculating (might take a while)..")
            self.threshold = MOODS.tools.threshold_from_p(self.pssm, self.bg, fpr)

        # Evaluating if motif is palindromic
        self.is_palindrome = [max(e) for e in self.pssm] == [max(e) for e in reversed(self.pssm)]


class Thresholds:
    """
    Container for all motif thresholds given default FPRs.

    Authors: Eduardo G. Gusmao.
    """

    def __init__(self, motif_data):
        """ 
        Initializes Thresholds. Motif thresholds are stored in a dictionary:
        [repository] -> [motif name] -> [fpr] -> threshold float

        Parameters:
        motif_data -- MotifData object.

        Variables:
        
        """

        # Initializing dictionary level 0
        self.dict = dict()

        # Iterating over fpr files
        for fpr_file_name in motif_data.get_fpr_list():

            # Initializing dictionary level 1
            fpr_name = ".".join(fpr_file_name.split("/")[-1].split(".")[:-1])
            self.dict[fpr_name] = dict()

            # Iterating in fpr file
            fpr_file = open(fpr_file_name, "r")
            header = fpr_file.readline()
            fpr_values = [float(e) for e in header.strip().split("\t")[1:]]
            for line in fpr_file:
                ll = line.strip().split("\t")
                # Initializing dictionary level 2
                self.dict[fpr_name][ll[0]] = dict()
                for i in range(1, len(ll)):
                    # Updating dictionary
                    self.dict[fpr_name][ll[0]][fpr_values[i-1]] = float(ll[i])
            fpr_file.close()
