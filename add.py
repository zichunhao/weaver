#!/usr/bin/env python                                                                                                                                                             
import os,glob
import argparse

def slice_it(li, cols=2):
    start = 0
    for i in range(0,cols):
        stop = start + len(li[i::cols])
        yield li[start:stop]
        start = stop

def files_toadd(filelist,size=1e10):
    isize = os.path.getsize(filelist[0])
    toadd = int(len(filelist)/(size/isize))
    print(toadd,len(filelist),isize/size)
    if len(filelist)>toadd and toadd>0:
        splitfiles = slice_it(filelist,toadd)
    else:
        splitfiles = [filelist]
    return splitfiles

def main(args):
    if args.sample:
        subfolders = [os.path.join(args.idir,args.sample+'/')]
    else:
        subfolders = glob.glob(args.idir+'/*/')
    for sub in subfolders:
        odir = os.path.join(args.odir,os.path.basename(os.path.normpath(sub)))
        filelist = sorted(glob.glob(sub+'/*/*/*/output_*.root'))
        splitfiles = files_toadd(filelist,1e10)
        os.system('mkdir -p %s'%odir)
        for isub,sublist in enumerate(splitfiles):
            toHadd = ' '.join([str(i0) for i0 in sublist])
            haddcommand = 'hadd -O -fk404 %s/haddoutput_%i.root %s'%(odir,isub,toHadd)
            if not (os.path.exists('%s/haddoutput_%i.root'%(odir,isub))):
                print(haddcommand)
                #os.system(haddcommand)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add files')
    parser.add_argument('--idir', default='/mnt/hadoop/store/user/jduarte/DNNTuples/train/', help='input dir')
    parser.add_argument('--odir', default='/data/shared/cmantill/DNNTuples/train/',help='output dir')
    parser.add_argument('--sample', default=None, help='hadd only this sample')
    args = parser.parse_args()
    main(args)
