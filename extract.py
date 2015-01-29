#!/usr/bin/env python

import argparse
import os
import errno
import shutil
import subprocess
import re
import multiprocessing as mp
from functools import partial
<<<<<<< HEAD
import itertools
=======
import time
# try:
#     sys.path.append('/slfs1/users/hedi7/utils/PYSGE')
# except:
#     print "No PYSGE found, cluster run support is removed"
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e

HTKTOOLSPATH = '/slfs1/users/yl710/htk/HTKTools/'


<<<<<<< HEAD
###
# Attention:
# All static variables which end with _DIR will be created by this script.
# If you want to implement your personal dirs, etc. keep in mind that everything
# ending with _DIR will be overwritten!
###

# The masked using to determinate which features will be used to perform normalization
# This mask represents a normalization for every utterance
MASK = '%%%%%%%%%%%%%%*'

QSUB = 'qsub'
=======
# The masked using to determinate which features will be used to perform normalization
# This mask represents a normalization for every utterance
MASK = "%%%%%%%%%%%%%%*"

>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e

ROOT_DIR = '.'
VAD_PATH = '/slfs1/users/hedi7/utils/FeatureExtract/vad/'
CONFIG_DIR = os.path.join(ROOT_DIR, 'cfgs')

HCOMPV = os.path.join(HTKTOOLSPATH, 'HCompV')
HCOPY = os.path.join(HTKTOOLSPATH, 'HCopy')
HEREST = os.path.join(HTKTOOLSPATH, 'HERest')
HHED = os.path.join(HTKTOOLSPATH, 'HHEd')
HPARSE = os.path.join(HTKTOOLSPATH, 'HParse')
HRESULTS = os.path.join(HTKTOOLSPATH, 'HResults')
HVITE = os.path.join(HTKTOOLSPATH, 'HVite')

VAD = os.path.join(VAD_PATH, 'vad')
VAD_GMM_CFG = os.path.join(VAD_PATH, 'gmm.cfg')
VAD_GMM_MMF = os.path.join(VAD_PATH, 'MMF')

FEATURES_DIR = os.path.join(ROOT_DIR, 'features')
STATIC_DIR = os.path.join(FEATURES_DIR, 'static')
DYNAMIC_DIR = os.path.join(FEATURES_DIR, 'concat')
<<<<<<< HEAD
CMVN_DIR = os.path.join(FEATURES_DIR, 'cmvn')
=======
CMVN_FEATURES = os.path.join(FEATURES_DIR, 'cmvn')
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e

LOG_DIR = os.path.join(ROOT_DIR, 'log')
TMP_DIR = os.path.join(ROOT_DIR, 'tmp')

CMEANDIR = os.path.join(TMP_DIR, 'cmn')
VARSCALEDIR = os.path.join(TMP_DIR, 'cvn')
FLIST_DIR = os.path.join(CONFIG_DIR, 'flists')
EDFILES_DIR = os.path.join(CONFIG_DIR, 'edfiles')

<<<<<<< HEAD
MLF_HEADER = '#!MLF!#'

NUMBER_JOBS = 20


def runbatch(argsbatch, cwd=os.getcwd(), logfiles=None):
    if logfiles:
        assert(len(logfiles) == len(argsbatch))
    import tempfile
    processes = []
    scriptfiles = []
    for args, logfile in zip(argsbatch, logfiles):
        qsubcmd = [QSUB]
        qsubcmd.extend('-P cpu.p'.split())
        qsubcmd.append('-cwd')
        qsubcmd.extend('-j y'.split())
        #We want to run the command with bash
        qsubcmd.extend('-S /bin/bash'.split())
        # we want to sync the processes so that the main process waits until it
        # finishes
        qsubcmd.extend('-sync y'.split())
        if logfile:
            qsubcmd.extend('-o {}'.format(logfile).split())
        scriptfile = tempfile.NamedTemporaryFile()
        scriptfile.write(" ".join(args))
        scriptfile.flush()
        qsubcmd.append(scriptfile.name)
        with open(os.devnull, 'w') as FNULL:
            processes.append(
                subprocess.Popen(qsubcmd, cwd=cwd, stdout=FNULL, stderr=subprocess.STDOUT))
        scriptfiles.append(scriptfile)
    for process, scriptfile in zip(processes, scriptfiles):
        process.wait()
        scriptfile.close()


def runlocal(argsbatch, cwd=os.getcwd(), logfile=None):
    '''
    Runs the given batch files in parallel locally with NUMBER_JOBS Processes
    '''
    try:
        pool = mp.Pool()
        pool.map(universal_worker, pool_args(execute, argsbatch, cwd, logfile))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        raise KeyboardInterrupt


def execute(args, cwd, log):
    try:
        if log:
            with open(log, 'w') as logp:
                subprocess.Popen(args, stdout=logp, stderr=logp, cwd=cwd).wait()
        else:
            subprocess.Popen(args, cwd=cwd).wait()
    except KeyboardInterrupt:
        raise KeyboardInterrupt


RUN_MODE = {
    'cluster': runbatch,
    'local': runlocal
}

global runner

=======
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e

def pprint(str, silent):
    if not silent:
        print str


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def create_dirs():
    dirstomake = {v for (k, v) in globals().items() if re.search('.+DIR', k)}
    map(mkdir_p, dirstomake)


def cleanup():
    shutil.rmtree(CONFIG_DIR)
    shutil.rmtree(TMP_DIR)


def readDir(input_dir):
    '''
    Reads from the given Inputdir recursively down and returns all files in the directories.
    Be creful since there is not a check if the file is of a specific type!
    '''
    foundfiles = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if os.path.isfile(os.path.join(root, f)):
                foundfiles.append(os.path.abspath(os.path.join(root, f)))
    return foundfiles


def generate_HCopy_script(files, output_features, output_script, featuretype=r'plp'):
    """
        Generate a script file for HCopy (mapping between 'audio' and feature files).
        files:     a list of all wav files
        outdir:     The output directory for the feature files
        output_script: Path of the output HCopy script.
    """
    with open(output_script, mode='w') as output:
        for f in files:
            fname = os.path.basename(os.path.splitext(f)[0])
            featureout = os.path.join(output_features, fname)
            output.write('{} {}.{}'.format(f, featureout, featuretype))
            output.write(os.linesep)


<<<<<<< HEAD
def splitintochunks(l, num):
    a = []
    spl, ext = divmod(len(l), num)
    for i in range(num):
        a.append(l[i * spl:(i + 1) * spl])
    # If he have a residual, we append the last entries into the last list
    if ext:
        a[-1].extend(l[-ext:])
    return a


=======
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
def splitScp(scpfile, chunksize=None):
    '''
    Splits the given scp file and returns a list of the new paths
    '''
<<<<<<< HEAD
    scplines = open(scpfile, 'r').read().splitlines()
    chunks = []
    if not chunksize:
        chunks = splitintochunks(scplines, NUMBER_JOBS)
=======
    def splitintochunks(l, num):
        a = []
        spl, ext = divmod(len(l), num)
        for i in range(num):
            a.append(l[i * spl:(i + 1) * spl])
        # If he have a residual, we append the last entries into the last list
        if ext:
            a[-1].extend(l[-ext:])
        return a

    scplines = open(scpfile, 'r').read().splitlines()
    chunks = []
    if not chunksize:
        chunks = splitintochunks(scplines, mp.cpu_count())
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    else:
        chunks = splitintochunks(scplines, chunksize)
    tardir = os.path.abspath(os.path.dirname(scpfile))
    basenamescp, ext = os.path.splitext(os.path.basename(scpfile))
    newfilepaths = []
    for i in range(len(chunks)):
        newfilename = "%s%i%s" % (basenamescp, i, ext)
        newfullfilepath = os.path.join(tardir, newfilename)
        newfilepaths.append(newfullfilepath)
        with open(newfullfilepath, 'w') as newfilep:
            for chunk in chunks[i]:
                newfilep.write(chunk)
                newfilep.write(os.linesep)
    return newfilepaths


<<<<<<< HEAD
# def multiprocessHCopy(scpfiles, config):
#     '''
#     Runs HCopy in multiple processes
#     '''
#     part = partial(HCopy, config)

#     threadpool = mp.Pool()

#     threadpool.map(part, scpfiles)
#     threadpool.close()
#     threadpool.join()

def parallelHCopy(scppaths, configpath):
    '''
    This is a helper method, since we use for the qsub -sync y
    '''
    argsbatch = []
    logfiles = []
    for scppath in scppaths:
        args = HCopy(configpath, scppath)
        stage = os.path.basename(os.path.splitext(scppath)[0])
        logfile = os.path.join(LOG_DIR, 'hcopy_{}.log'.format(stage))
        logfiles.append(logfile)
        argsbatch.append(args)
    global runner
    runner(argsbatch, os.getcwd(), logfiles)


def writeOutSplits(splits, outputdir, outputname):
    '''
    Writes out the splits into N different files and returns the list of files
    '''
    ret = []
    for i in range(len(splits)):
        newoutname = "%s_%i" % (outputname, i)
        outfile = os.path.join(outputdir, newoutname)
        ret.append(outfile)
        with open(outfile, 'w') as outpointer:
            outpointer.writelines(splits[i])
    return ret


def HCopy(config, hcopy_scp):

=======
def multiprocessHCopy(scpfiles, config):
    '''
    Runs HCopy in multiple processes
    '''
    part = partial(HCopy, config)

    threadpool = mp.Pool()
    ts = time.time()
    threadpool.map(part, scpfiles)
    te = time.time()
    print "Multi : %.2f" % (te - ts)
    threadpool.close()
    threadpool.join()


def HCopy(config, hcopy_scp):
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    args = []
    args.append(HCOPY)  # HCopy binary
    args.extend(r'-T 1'.split())
    # Config file
    args.extend(r'-C {}'.format(os.path.join(config)).split())
    args.extend(r'-S {}'.format(hcopy_scp).split())  # Script file

<<<<<<< HEAD
    return args
    # with open(os.path.join(LOG_DIR, 'hcopy_{}.log'.format(stage)), mode='w') as log:
    #     subprocess.check_call(
    #         args,  stdout=log, stderr=subprocess.STDOUT)
=======
    stage = os.path.basename(os.path.splitext(hcopy_scp)[0])

    with open(os.path.join(LOG_DIR, 'hcopy_{}.log'.format(stage)), mode='w') as log:
        subprocess.check_call(
            args,  stdout=log, stderr=subprocess.STDOUT)
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e


def generate_HCompV_script(input_dir, output_script, featuretype='plp'):
    """
        Generate a script file for HCompV (list of 'mfcc' files).
        input_dir:     Directory containing the audio files.
        output_script: Path of the output HCompV script.
    """
    with open(output_script, mode='w') as output:
        files = (f for f in sorted(os.listdir(input_dir))
                 if os.path.isfile(os.path.join(input_dir, f)))
        for f in files:
            feature = os.path.join(FEATURES_DIR, f.split('.')[0])
            output.write('{}.featuretype'.format(feature))
            output.write(os.linesep)


def HCompV(train, config, readdir=None, floor=None, proto=None, mask=None, hmmflatdir=None, computemeans=None, outputproto=None, clusterrequest=None, clusterdir=None):
    '''
    train: path to scp file with the input training data
    config: path to the config file which will be used for the estimation
    (opt)readdir : uses the -c parameter
    (opt)proto : Path to the prototype file
    (opt)hmmflatdir : reads in an already generated flat dir, for flat start
    (opt)computemeans : only computes the means
    (opt)outputproto: Outputs a newly generated flat model into that given path
    (opt)clusterrequest : [mnv] includes in the output either mean(m option), variance (v) or number of frames (n)
    (opt)clusterdir: Calculates the specific Mean/variance and stores in the given dir
    '''
    args = []
    args.append(HCOMPV)  # HCompV binary
    args.append('-A')
    args.append('-D')
    args.append('-V')
    args.extend('-C {}'.format(config).split())  # Config file
    # Generate variance floor macro named 'vFloors'
    if floor:
        args.extend('-f {}'.format(floor).split())
    # Output folder for 'vFloors'
    if hmmflatdir:
        args.extend('-M {}'.format(hmmflatdir).split())
    if mask:
        args.extend('-k {}'.format(mask).split())
    if computemeans:
        args.extend('-m')
    if readdir:
        args.extend('-c {}'.format(readdir).split())
    # New version of proto
    if outputproto:
        args.extend('-o {}'.format(outputproto).split())
    if clusterrequest:
        args.extend('-q {}'.format(clusterrequest).split())
    if proto:
        args.append('{}'.format(proto))  # Prototype file
    if clusterdir:
        args.extend('-c {}'.format(clusterdir).split())
    args.extend('-S {}'.format(train).split())  # List of training files (mfcc)
<<<<<<< HEAD

    return args
    # global runner
    # TODO: Change it to cluster, but Cluster submit does not work for some
    # reason
    # runner(args, logfile=os.path.abspath(os.path.join(
    #     LOG_DIR, 'hcompv_{}.log'.format(trainname))))
=======
    trainname = os.path.splitext(os.path.basename(train))[0]
    with open(os.path.join(LOG_DIR, 'hcompv_{}.log'.format(trainname)), mode='w') as log:
        subprocess.Popen(args, stdout=log).wait()
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e


def concatenateSpeech(vadmlf, featuretype='plp'):
    '''
    Generates a file which consists of HCopy Commands to concatenate the speech segments and removing therefore the silenced segments
    vadmlf : the VAD.mlf which was generated using the VAD command
    '''
<<<<<<< HEAD
    commands = []
    with open(vadmlf, 'r') as lines:
        for line in lines:
            # remove newline
            line = line.rstrip(os.linesep)
=======
    running_concats = []
    with open(vadmlf, 'r') as lines:
        for line in lines:
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
            # features and labels are indicated as "*/XXXXXXXXXX.lab"
            if line.startswith("\""):
                # We know that the first 3 characters are "*/, so we remove
                # these and the last trailing "
                label = line[3:-2]
                # The label is know in the form of XXXXXXXXXXXX.lab
                curlabel, _ = label.split(".")
                # Remember which current label we got
                # Find all speech segments for the current label
                newline = lines.next()
                speechsegments = []
                while re.match("^[0-9]{5,}", newline):
                    begin, end, segmentflag = newline.split()
                    begin = long(begin) / 100000
                    end = long(end) / 100000
                    if segmentflag == 'speech':
                        speechsegments.append((begin, end))
                    newline = lines.next()
<<<<<<< HEAD
                # If we have non-silent speech
                if speechsegments:
                    log = os.path.join(LOG_DIR, 'concat.log')
                    cmd = []
                    cmd.append(HCOPY)
                    cmd.extend('-T 1'.split())
                    for i in range(len(speechsegments)):
                        begin, end = speechsegments[i]
                        cmd.append(
                            "{}.{}[{},{}]".format(curlabel, featuretype, begin + 1, end - 1))
                        if i + 1 < len(speechsegments):
                            cmd.extend("+")
                    featurename = curlabel + '.' + featuretype
                    cmd.append(
                        os.path.abspath(os.path.join(DYNAMIC_DIR, featurename)))
                    commands.append(cmd)
                    with open(log, 'a') as logp:
                        subprocess.Popen(
                            cmd, cwd=STATIC_DIR, stdout=logp, stderr=logp).wait()

        # for cmd in commands:
        #     cmd.wait()

    # global runner
    # commandchunks = splitintochunks(commands, 2)
    # logchunks = splitintochunks(logs, 2)
    # for commandchunk, logchunk in zip(commandchunks, logchunks):
    #     runner(commandchunk, os.getcwd(), logchunk)
    # for concat in running_concats:
    #     concat.wait()
=======
                cmd = []
                cmd.append(HCOPY)
                for i in range(len(speechsegments)):
                    begin, end = speechsegments[i]
                    cmd.append(
                        "{}.{}[{},{}]".format(curlabel, featuretype, begin + 1, end - 1))
                    if i + 1 < len(speechsegments):
                        cmd.extend("+")
                featurename = curlabel + '.' + featuretype
                cmd.append(
                    os.path.abspath(os.path.join(DYNAMIC_DIR, featurename)))
                running_concats.append(
                    subprocess.Popen(cmd, cwd=os.path.abspath(
                        STATIC_DIR)))
    for concat in running_concats:
        concat.wait()
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e


def writeSimpleScp(files, outputpath):
    with open(outputpath, 'w') as scppointer:
        for f in files:
            scppointer.write(f)
            scppointer.write(os.linesep)


<<<<<<< HEAD
def parallelVad(scpfile, outputmlf):
    # Generate one "base" .scp file which will then be split up
    scpsplits = splitScp(scpfile)
    outputmlfs = []
    argsbatch = []
    logfiles = []
    for i in range(len(scpsplits)):
        outputMLF = os.path.abspath(
            os.path.join(TMP_DIR, 'vad_%i.mlf' % (i + 1)))
        logfile = os.path.abspath(
            os.path.join(LOG_DIR, 'vad_%i.log' % (i + 1)))
        outputmlfs.append(outputMLF)
        # Get the current arguments for the VAD
        args = vad(scpsplits[i], outputMLF)
        argsbatch.append(args)
        logfiles.append(logfile)

    cwdpath = VAD_PATH
    global runner
    runner(argsbatch, cwdpath, logfiles)

    # Collect all the results, the only problem is the #!MLF!# header in each
    # file
    collectmlf = []
    for mlf in outputmlfs:
        collectmlf.extend(open(mlf, 'r').read().splitlines()[1:])
        os.remove(mlf)
    with open(outputmlf, 'w') as outputmlfpointer:
        outputmlfpointer.write(MLF_HEADER)
        outputmlfpointer.write(os.linesep)
        outputmlfpointer.writelines(os.linesep.join(collectmlf))


def universal_worker(input_pair):
    function, args = input_pair
    return function(*args)


def pool_args(function, *args):
    args, cwd, logs = args
    # make sure we have enough args
    a = [cwd for i in args]
    return zip(itertools.repeat(function), zip(args, a, logs))


def vad(scpfile, outputmlf):
    # global runner
    '''
    scpfile: The scp file which contains the utterances to do VAD
    outputmlf : Path to the file which will be generated after VAD
    returns the arguments which will be executed in the form of (args,cwd,logfile)
    '''
    # Generate the .scp file
    # scpfile = os.path.abspath(os.path.join(TMP_DIR, 'vad.scp'))
    # writeSimpleScp(wavfiles, scpfile)
    args = []
    # outputmlf = os.path.abspath(outputmlf)
=======
def vad(wavfiles, outputdir):
    '''
    wavfiles: the absolute filepaths to the given wave files
    outputdir : directory where the concatenated and silence removed features will be stored
    '''
    # Generate the .scp file
    scpfile = os.path.abspath(os.path.join(TMP_DIR, 'vad.scp'))
    writeSimpleScp(wavfiles, scpfile)
    args = []
    resultMLF = os.path.abspath(os.path.join(TMP_DIR, 'vad.mlf'))
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    # WE need to use everywhere an absolute path , since we will run the .vad in
    # it's directory, meaning that every realtive link will fail
    args.append(VAD)
    args.append(scpfile)
    args.append(VAD_GMM_CFG)
    args.append(VAD_GMM_MMF)
<<<<<<< HEAD
    args.append(os.path.abspath(outputmlf))
    # For the VAD tool we need to run it in the given folder otherwise ./HList
    # will not be found, so we use cwd = "" to do so
    return args
    # runner(args, cwd=VAD_PATH, logfile=os.path.abspath(
    #     os.path.join(LOG_DIR, 'vad.log')))
    # with open(os.path.join(LOG_DIR, 'vad.log'), 'w') as log:
    #     subprocess.Popen(
    #         args, stdout=log, stderr=subprocess.STDOUT, cwd=VAD_PATH).wait()
=======
    args.append(resultMLF)
    # For the VAD tool we need to run it in the given folder otherwise ./HList
    # will not be found, so we use cwd = "" to do so
    with open(os.path.join(LOG_DIR, 'vad.log'), 'w') as log:
        subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, cwd=VAD_PATH).wait()
    return resultMLF
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e


def readFeatureConfig(config):
    conf = {}
    for line in config:
        confline = re.findall(r"[\w\.']+", line)
        if confline:
            # Remove the trailing HPARM in the HPARAM
            if confline[0] == 'HPARM':
                confline = confline[1:]
            param, value = confline
            conf[param] = value
    return conf


def calculate_target_dimension(config, targetkind):
    '''
    Calculates the target dimension given the targetkind. We read in the targetkind
    and parse out the relevant parameters "A","0","D","T"
    targetkind : the resulting target feature, e.g.
    '''
    staticdimension = 0
    # Get the config for the size of cepstral compoentns
    staticdimension += int(config['NUMCEPS'])
    feature = re.split('[_ -]', targetkind)
    # Use the UPPER case letters, for convienience
    feature = map(lambda x: x.upper(), feature)
    # First check if we also append C0 Energy, meaning that the static feature
    # has size +1
    if "0" in feature:
        staticdimension += 1

    targetdimension = staticdimension
    if "A" in feature:
        targetdimension += staticdimension
    if "D" in feature:
        targetdimension += staticdimension
    if "T" in feature:
        targetdimension += staticdimension
    return targetdimension


def checkvalidfeatures(featureconfig, targetkind):
    '''
    Checks if the given features are valid, e.g. Statis features are PLP_0 and dynamic are PLP_0_D_A_Z
    Since no conversion between features are possible we simply check if the two feature types (PLP,PLP) are equal
    '''
    statickind = featureconfig['TARGETKIND']

    statictype = re.split('[_ -]', statickind)[0]
    targettype = re.split('[_ -]', targetkind)[0]
    # Raise an error if statictype is empry or targettype, meaning that there
    # was no match
    if not statictype or not targettype or not targettype.lower() == statictype.lower():
        raise ValueError('The specified Targetkind for the static features (%s) is not equal to the one for the dynamic features (%s)' % (
            statictype, targettype))


def generate_cut_cmn_cvn_config(config, targetkind, targetdimension):
    '''
    config is the read out feature config.
    This function does pare the current config and returns a tuple in the form of:
    (cut,cmn,cvn) files which are all paths

    '''

    globvar = os.path.join(CONFIG_DIR, 'globvar')
    cmnconfig = os.path.join(CONFIG_DIR, 'cmn.cfg')
    cvnconfig = os.path.join(CONFIG_DIR, 'cvn.cfg')
    cutconfig = os.path.join(CONFIG_DIR, 'cut.cfg')
    cmntargetkind = config['TARGETKIND']

    # First process cmn.cfg
    with open(cmnconfig, 'w') as cmnpointer:
        cmnpointer.write("TARGETKIND = {}".format(cmntargetkind))
        cmnpointer.write(os.linesep)
        cmnpointer.write("TRACE = {}".format(1))
        cmnpointer.write(os.linesep)
        cmnpointer.write("MAXTRYOPEN = {}".format(1))
    with open(cvnconfig, 'w') as cvnpointer:
        cvnpointer.write("TARGETKIND = {}".format(targetkind))
        cvnpointer.write(os.linesep)
        cvnpointer.write("TRACE = {}".format(1))
        cvnpointer.write(os.linesep)
        cvnpointer.write("MAXTRYOPEN = {}".format(1))
        cvnpointer.write(os.linesep)
        cvnpointer.write(
            "CMEANMASK = {}/{}".format(os.path.abspath(DYNAMIC_DIR), MASK))
        cvnpointer.write(os.linesep)
        cvnpointer.write("CMEANDIR = {}".format(CMEANDIR))
    with open(cutconfig, 'w') as cutpointer:
        cutpointer.write("TARGETKIND = {}".format(targetkind))
        cutpointer.write(os.linesep)
        cutpointer.write("TRACE = {}".format(1))
        cutpointer.write(os.linesep)
        cutpointer.write("MAXTRYOPEN = {}".format(1))
        cutpointer.write(os.linesep)
        cutpointer.write(
            "CMEANMASK = {}/{}".format(os.path.abspath(DYNAMIC_DIR), MASK))
        cutpointer.write(os.linesep)
        cutpointer.write("CMEANDIR = {}".format(CMEANDIR))
        cutpointer.write(os.linesep)
        cutpointer.write(
            "VARSCALEMASK = {}/{}".format(os.path.abspath(DYNAMIC_DIR), MASK))
        cutpointer.write(os.linesep)
        cutpointer.write("VARSCALEDIR = {}".format(VARSCALEDIR))
        cutpointer.write(os.linesep)
        cutpointer.write("VARSCALEFN = {}".format(globvar))
    with open(globvar, 'w') as globpointer:
        globpointer.write("<VARSCALE> {}".format(targetdimension))
        globpointer.write(os.linesep)
        for i in range(targetdimension):
            globpointer.write("%.1f " % (i))
    return (cutconfig, cmnconfig, cvnconfig)


def cmvn(cutcfg, cmncfg, cvncfg):
    '''
    cutcfg: Path to the cut config file
    cmncfg: Path to the cmn config file
    cvncfg: Path to the cvn config file
    returns path to the resulting .scp file
    '''
<<<<<<< HEAD
    global runner
    concat_features = readDir(DYNAMIC_DIR)
    norm_script = os.path.abspath(
        os.path.join(TMP_DIR, '{}.scp'.format('norm')))
=======
    concat_features = readDir(DYNAMIC_DIR)
    norm_script = os.path.join(TMP_DIR, '{}.scp'.format('norm'))
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    data_scp = os.path.join(TMP_DIR, '{}.scp'.format('data'))
    # Write out the general data file
    writeSimpleScp(concat_features, data_scp)
    # And the HCopy file

<<<<<<< HEAD
    generate_HCopy_script(concat_features, CMVN_DIR, norm_script)
    # Run first the cmn HCOMPV
    # The mask needs to be with ABSPATH otherwise it wouldn't work since
    # The given .scp file consists of absolute paths
    FEATUREMASK = os.path.join(os.path.abspath(DYNAMIC_DIR), MASK)
    # Mean normalization

    scpsplits = splitScp(data_scp)
    mean_normalization_args = []
    variance_normalization_args = []
    logfiles = []
    for scpsplit in scpsplits:
        mean_args = HCompV(scpsplit, cmncfg, mask=FEATUREMASK,
                           clusterrequest='m', clusterdir=CMEANDIR)
        variance_args = HCompV(data_scp, cvncfg, mask=FEATUREMASK,
                               clusterrequest='v', clusterdir=VARSCALEDIR)
        mean_normalization_args.append(mean_args)
        variance_normalization_args.append(variance_args)

        trainname = os.path.splitext(os.path.basename(scpsplit))[0]
        logfile = os.path.abspath(os.path.join(
            LOG_DIR, 'hcompv_{}.log'.format(trainname)))
        logfiles.append(logfile)

    # first do mean normalization
    runner(mean_normalization_args, os.getcwd(), logfiles)

    # variance normalization
    runner(variance_normalization_args, os.getcwd(), logfiles)

    # normsplits = splitScp(norm_script)
    # multiprocessHCopy(normsplits, cutcfg)

    normsplits = splitScp(norm_script)

    parallelHCopy(normsplits, cutcfg)

    # args = HCopy(cutcfg, norm_script)
    # logfile = os.path.join(LOG_DIR, 'HCopy_cut.log')
    # runner([args], os.getcwd(), [logfile])

    cmnv_features = readDir(CMVN_DIR)
=======
    generate_HCopy_script(concat_features, CMVN_FEATURES, norm_script)
    # Run first the cmn HCOMPV
    FEATUREMASK = os.path.join(os.path.abspath(DYNAMIC_DIR), MASK)
    # Mean normalization
    HCompV(data_scp, cmncfg, mask=FEATUREMASK,
           clusterrequest='m', clusterdir=CMEANDIR)
    # variance normalization
    HCompV(data_scp, cvncfg, mask=FEATUREMASK,
           clusterrequest='v', clusterdir=VARSCALEDIR)
    normsplits = splitScp(norm_script)
    multiprocessHCopy(normsplits, cutcfg)
    # HCopy(cutcfg,norm_script)

    cmnv_features = readDir(CMVN_FEATURES)
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    cmnv_scp = os.path.join(TMP_DIR, 'cmvn.scp')
    writeSimpleScp(cmnv_features, cmnv_scp)
    return cmnv_scp


<<<<<<< HEAD
def extractstaticFeatures(wavfiles, configpath):
    '''
    wavfiles : list of the wavfiles which will be processed
    configpath: String, filepath to the configure file for HCopy
    '''

    hcopy_scp = os.path.join(TMP_DIR, '{}.scp'.format('static'))
    # Extracting the static features
    generate_HCopy_script(wavfiles, STATIC_DIR, hcopy_scp)
    scpsplits = splitScp(hcopy_scp)
    # splitpaths = writeOutSplits(scpsplits, TMP_DIR, 'hcopy_scp')
    parallelHCopy(scpsplits, configpath)


def audio_scp_type(value):
    '''
    This function is the type of an argparse parameter
    It is used to distinguish if the program should extract the static features
    or if that is already done.
    If the given parameter value ends with .scp, we will treat the file as an scp file
    thus running the whole feature extraction process
    '''

    if value.endswith('.scp'):
        return open(value, 'r').read().splitlines()
    else:
        return readDir(value)

=======
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
if __name__ == "__main__":
    """
        Feature extraction script
    """

    parser = argparse.ArgumentParser(description='Feature extraction using multiple processes on the machine'
                                     )

<<<<<<< HEAD
    parser.add_argument('sourceAudio',
                        type=audio_scp_type, help='The root directory of all .wav files or an already feature extracted .scp file')
=======
    parser.add_argument('rootaudiodir',
                        type=str, help='The root directory of all .wav files')
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    parser.add_argument(
        '-c', '--config', dest='config', type=argparse.FileType('r'), help='HTK Config file for the feature exraction', required=True
    )

    parser.add_argument(
        '-t', '--targetkind', help='The Feature which will be created at least, e.g. PLP_0_D_A_Z', required=True, type=str)

    parser.add_argument('--silent', action='store_true',
                        help='Disable the progression messages'
                        )

    parser.add_argument('--clean', action='store_true',
                        help='Cleanup the generated files after processing'
                        )

<<<<<<< HEAD
    parser.add_argument(
        '--run', help='runs on either Cluster or locally the extraction job', default='local')
    parser.add_argument(
        '--vadmlf', type=str, help='If Vad was already done, mlf file can be provided, so that vad will not be run')

=======
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    args = parser.parse_args()

    ### Step 0 - Setup the Working Directories ###

    pprint('Setup the Working Directories', args.silent)
    create_dirs()

<<<<<<< HEAD
    global runner
    runner = RUN_MODE[args.run]

=======
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    pprint('Reading in Config ', args.silent)
    config = readFeatureConfig(args.config)
    # Check if the given features are compatible
    targetkind = args.targetkind
    # Targetkind validation
    # Convert targetkind to upper case
    targetkind = "".join(map(lambda x: x.upper(), targetkind))
    # replace all wrong possible delimitors
    targetkind = re.sub("[ -]", "_", targetkind)

    checkvalidfeatures(config, targetkind)
    targetdimension = calculate_target_dimension(config, targetkind)
    configpath = args.config.name

    ### Step 1 - Extract the Static Features ###

<<<<<<< HEAD
    wavfiles = args.sourceAudio

    extractstaticFeatures(wavfiles, configpath)
=======
    pprint('Extracting the features', args.silent)

    wavfiles = readDir(args.rootaudiodir)

    hcopy_scp = os.path.join(TMP_DIR, '{}.scp'.format('static'))
    # Extracting the static features
    generate_HCopy_script(wavfiles, STATIC_DIR, hcopy_scp)
    scpsplits = splitScp(hcopy_scp)
    multiprocessHCopy(scpsplits, configpath)
    ts = time.time()
    HCopy(configpath, hcopy_scp)
    te = time.time()
    print "Single : %2.f" % (te - ts)
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e

    ### Step 2 - VAD ###

    pprint('Running VAD', args.silent)
<<<<<<< HEAD
    if args.vadmlf:
        vadmlf = args.vadmlf
    else:
        vadscpfile = os.path.join(TMP_DIR, 'vad.scp')
        writeSimpleScp(wavfiles, vadscpfile)
        vadmlf = os.path.join(TMP_DIR, 'vad.mlf')
        parallelVad(vadscpfile, vadmlf)

    # vadmlf = os.path.join(ROOT_DIR, 'vad_spoof.mlf')

=======
    vadmlf = vad(wavfiles, DYNAMIC_DIR)
    # vadmlf = os.path.join(TMP_DIR, 'vad.mlf')
>>>>>>> 3640e5f3eef6aab22ade8fa822536eae4ba39b1e
    pprint('Concatinating Frames', args.silent)
    concatenateSpeech(vadmlf)

    ### Step 3 - CMVN ####
    cut, cmn, cvn = generate_cut_cmn_cvn_config(
        config, targetkind, targetdimension)
    pprint('Running CMVN', args.silent)
    cmvn(cut, cmn, cvn)

    pprint('Done !', args.silent)
    if args.clean:
        cleanup()
