#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import re
import os
from netCDF4 import Dataset
import sys
import codecs
from copy import deepcopy as dc

np.set_printoptions(threshold = np.inf) # ensure print whole array

class test():
    def __init__(self):


        self.nyOut = 360
        self.nxOut = 720
        self.nlat = 360
        self.nlon = 720
##

    def landIndex(self,fnuse1,fnuse2):
# Land index for GCPCH4 2021 (GSWP3/CRUJRA) grid:
 
#        print(fnuse1)
        fid = Dataset( fnuse1, 'r')
        self.lon2d = (fid.variables['lon2d'][:,:])
        self.lat2d = (fid.variables['lat2d'][:,:])
        self.lsmask = (fid.variables['land_sea_mask'][:,:])
        fid.close()
        self.nx = len(self.lon2d[0,:])
        self.ny = len(self.lat2d[:,0])
        self.lon1d = dc(self.lon2d[0,:])
        self.lat1d = dc(self.lat2d[:,0])

        fid = Dataset( fnuse2, 'r')
        self.lonland = (fid.variables['longitude'][:,:])
        self.latland = (fid.variables['latitude'][:,:])

        fld_land = (fid.variables['t1p5m_gb'][:,:,:])[0,:,:]
        fid.close()

        fld_land = fld_land.flatten()
        self.lonland = (self.lonland.flatten())
        self.latland = (self.latland.flatten())
       
        self.nland = np.sum((self.lsmask > 0))
        self.lat = self.lat2d[(self.lsmask > 0)]
        self.lon = self.lon2d[(self.lsmask > 0)]

        self.lsmask_cru = dc(self.lsmask)
        for l in range(self.nland):
            valx = np.where(self.lon1d == self.lonland[l])
            valy = np.where(self.lat1d == self.latland[l])
            if (np.sum(len(valx)) > 0):
                if(np.float32(np.isfinite(fld_land[l])) <= 0):
                    self.lsmask_cru[valy[0],valx[0]] = 0

# set up index for conversion from land only to global:
        self.lon_index = np.zeros(self.nland,dtype = np.int64)
        self.lat_index = np.zeros(self.nland,dtype = np.int64)

        maskflat = dc(self.lsmask).flatten()
        lon2dflat = dc(self.lon2d).flatten()
        lat2dflat = dc(self.lat2d).flatten()

        ic = -1
        for l in range(self.nland):
            valx = np.where(self.lon1d == self.lonland[l])
            valy = np.where(self.lat1d == self.latland[l])
            if (np.sum(len(valx)) > 0):
                ic = ic + 1
                self.lon_index[ic] = valx[0]
                self.lat_index[ic] = valy[0]

        print('converted land mask')
##
##

    def readFldFixed2d(self, fn, mdi, fld):

        fid = Dataset( fn, "r")
        fld = fid.variables[fld][:,:]
        fid.close()

        fld2d = np.ma.masked_array( fld, fld < 0)
 
        self.fld2d = fld2d
##

    def readFld(self, fn, mdi, im, fld):

        fid = Dataset( fn, "r")
        fld = fid.variables[fld][im,:,:]# take im month
        fid.close()

        fld2d = np.zeros([self.ny,self.nx],dtype = np.float32)
        fld2d[:,:] = mdi
        fld1 = fld[0,:]

        for l in range(self.nland):
            fld2d[self.lat_index[l],self.lon_index[l]] = fld1[l]
        fld2d = np.ma.masked_array( fld2d, fld2d <= mdi)
 
        self.fld2d = fld2d
        self.fld = fld

##

    def readFldMult(self, fn, mdi, im, fld, nd):

        fid = Dataset( fn, "r")
        fld = fid.variables[fld][im,nd,:,:]# take im month and nd
        fid.close()

        fld2d = np.zeros([self.ny,self.nx],dtype = np.float32)
        fld2d[:,:] = -10.
        fld1 = fld[0,:]

        for l in range(self.nland):
            fld2d[self.lat_index[l],self.lon_index[l]] = fld1[l]
        fld2d = np.ma.masked_array( fld2d, fld2d <= mdi)
 
        self.fld2d = fld2d
        self.fld = fld

##

    def getAreaWgt(self):
 
        dlat = self.lat2d[1,0]-self.lat2d[0,0]
        dlon = self.lon2d[0,1]-self.lon2d[0,0]
        print('dlat,dlon',dlat,dlon)

        ## Area weighting taken from doc of iris.analysis.cartography. But should be sine not cosine:
        #        radius_earth = 6.371e6
        #        area = radius_earth*radius_earth*(np.radians(dlon))*(np.sin(np.radians(self.lat2d+0.5*dlat))-np.sin(np.radians(self.lat2d-0.5*dlat)))
        
        #        wgt = np.radians(dlon)*(np.sin(np.radians(self.lat2d+0.5*dlat))-np.sin(np.radians(self.lat2d-0.5*dlat)))
   
        # define oblate spheroid from WGS84
        a = 6378137
        b = 6356752.3142
        e2 = 1 - (b**2/a**2)
        lat = np.radians(self.lat2d+0.5*dlat)
        lat_gc = np.arctan( (1-e2)*np.tan(lat) )

        # radius equation
        # see equation 3-107 in WGS84
        radius_earth_acc = (
            (a * (1 - e2)**0.5) 
            / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )
    
        #        wgt = radius_earth_acc*radius_earth_acc*np.radians(dlon)*np.radians(dlat)*(np.cos(np.radians(self.lat2d+0.5*dlat)))
        wgt = (np.radians(dlon))*(np.sin(np.radians(self.lat2d+0.5*dlat))-np.sin(np.radians(self.lat2d-0.5*dlat)))
        wgt = np.absolute(wgt)

        area = radius_earth_acc*radius_earth_acc*(np.radians(dlon))*(np.sin(np.radians(self.lat2d+0.5*dlat))-np.sin(np.radians(self.lat2d-0.5*dlat)))
        area = np.absolute(area)
        

        self.wgt2d = wgt
        self.area2d = area

##
    def read_global_dai_rflow_obs(self,nso,mon_len,mon_lenl):

        fn = 'coastal-stns-Vol-monthly.updated-Aug2014.nc'
        fid = Dataset( fn, "r")
        self.date = np.array(fid.variables["time"])
        self.lon = np.array(fid.variables["lon"])
        self.lat = np.array(fid.variables["lat"])
        self.area = np.array(fid.variables["area_stn"])
        self.alt = np.array(fid.variables["elev"])
        self.river = np.array(fid.variables["riv_name"])
        self.rflow = np.array(fid.variables["FLOW"])
        stn_name = np.array(fid.variables["stn_name"])
        ct_name = np.array(fid.variables["ct_name"])
        fid.close()

        riv_name=dc(self.river)


        criv_name_list=[]
        criv_name=np.chararray(len(riv_name[:,0]), itemsize=len(riv_name[0,:]))
        for i in range(len(riv_name[:,0])):
            criv_name[i] = (''.join(str(riv_name[i,:])))  
            tmp_name = []
            nc = np.int32(riv_name.shape[1])
            for c in range(nc):
                tmp_name.append(str(riv_name[i,c].decode('ISO-8859-1')))
            criv_name_list.append(''.join(tmp_name))       
#        print('river_name_list',criv_name_list)

        cct_name=np.chararray(len(ct_name[:,0]), itemsize=len(ct_name[0,:]))
        for i in range(len(ct_name[:,0])):
            cct_name[i] = ''.join(str(ct_name[i,:]))       

        cstn_name_list=[]
        cstn_name=np.chararray(len(stn_name[:,0]), itemsize=len(stn_name[0,:]))
        for i in range(len(stn_name[:,0])):
            cstn_name[i] = (''.join(str(stn_name[i,:])))  
            tmp_name = []
            nc = np.int32(stn_name.shape[1])
            for c in range(nc):
                tmp_name.append(str(stn_name[i,c].decode('ISO-8859-1')))
            cstn_name_list.append(''.join(tmp_name))       
#            print('cstn_name_list',cstn_name_list)

        year = (self.date/100)
        for i in range(len(year)):
            year[i] =np.int32(year[i])
        mon = self.date-year*100 
            
        self.station = dc(cstn_name)
        self.station_list = dc(cstn_name_list)
        self.river_list = dc(criv_name_list)
        self.country = dc(cct_name)

        self.rflow = np.transpose(self.rflow)

#mon_len,mon_lenl,sndays_from_1janor):
        stationor = self.station
        stationor_list = self.station_list
        lator = self.lat
        lonor = self.lon
        areaor = self.area
        altor = self.alt
        srflowor = self.rflow

        ntime=len(self.date)
        sdateor = np.zeros((nso,ntime),dtype = np.int64)
        syearor = np.zeros((nso,ntime),dtype = np.int64)
        smonor = np.zeros((nso,ntime),dtype = np.int64)

        for i in range(nso):
            sdateor[i,:] = self.date[:]
            syearor[i,:] = year[:]
            smonor[i,:] = mon[:]

        self.sdate = sdateor
        self.syear = syearor
        self.smon = smonor

##
##
    def doStuff(self):
##

#        l_bug = 1 # true if jules_vn4.7 and before. Not sure about 4.8
        l_bug = 0 # true if jules_vn4.7 and before. Not sure about 4.8

# Related to figure format:
        fraction = 0.042
        pad = 0.10
        nrows = 5
#        ncols = 1
        ncols = 2
        nsplots = nrows*ncols
        lstyle=['solid','dashed','dotted']
        lthin = 0.5
        month = [1,2,3,4,5,6,7,8,9,10,11,12]

        
        mon_len = [31,28,31,30,31,30,31,31,30,31,30,31]
        mon_lenl = [31,29,31,30,31,30,31,31,30,31,30,31]


#-----------------------------------------------------
# Read in obs river flow:
        nso = 925 # number of gauging stations in file
        self.read_global_dai_rflow_obs(nso,mon_len,mon_lenl)

        stationor = self.station
        stationor_list = self.station_list
        lator = self.lat
        lonor = self.lon
        areaor = self.area
        altor = self.alt
        rflowor = self.rflow
        dateor = self.sdate
        yearor = self.syear
        monor = self.smon

# initialise:
        stationo = []        
        lato = np.zeros([nso],dtype = np.float32)
        lono = np.zeros([nso],dtype = np.float32)
        alto = np.zeros([nso],dtype = np.float32)
        areao = np.zeros([nso],dtype = np.float32)
        
        rflowo = np.zeros([nso,1380],dtype = np.float32)        # random guess at max number of timesteps and sites
        ndays_from_1jano = np.zeros([nso,1380],dtype = np.float32) 
        yearo = np.zeros([nso,1380],dtype = np.float32)
        mono = np.zeros([nso,1380],dtype = np.float32) 
        dateo = np.zeros([nso,1380],dtype = np.float32)

        print('nso',nso,stationor.shape,rflowor.shape)
        for ist in range(nso):
            stationo.append(stationor[ist])
            
        rflowo[0:nso,:] = dc(rflowor[0:nso,:])
        lato = dc(lator[0:nso])
        lono = dc(lonor[0:nso])
        alto = dc(altor[0:nso])
        areao = dc(areaor[0:nso])
        yearo[0:nso,:] = dc(yearor[:])
        mono[0:nso,:] = dc(monor[:])
        dateo[0:nso,:] = dc(dateor[:])

        nsoo = len(stationo)

###
# now read in modelled data:

# Modelled river flow:
#        expt = 'u-ck843'      # GSWP3 oxisols+ultisols prognostic fsat
#        expt = 'u-ck844'      # CRUJRA oxisols+ultisols prognostic fsat
        expt_all = ['u-bk886']  # ['u-ck843','u-ck844']
        nex = len(expt_all)
#------------------------------------------------------
        nm = 12
        mdi = -1.e30
        rescale = 1.   # for testing purposes


        for ie in range(nex):
            expt = expt_all[ie]
            print('expt',expt)
#           Lat lon mask:
            fn1 = '/project/jchmr/hadng/data/d00/hadea/isimip3a/jules_ancils/landseamask/mask_latlon2d.nc'

            fn_expt = '/CRUJRA_V2P2P5D/isimip3a_notriffid_crujra_v2p2p5d_historical.ilamb.'
            fn_expt2 = '/CRUJRA_V2P2P5D/isimip3a_notriffid_crujra_v2p2p5d_historical.gen_mon_gb.'
            output_type1 = 'CRU_'
#            yre=2020
            l_gswp3 = 0
            if((expt == 'u-ck348') | (expt == 'u-ck594') | (expt == 'u-ck769') \
               | (expt == 'u-ck843')  | (expt == 'u-ck917') | (expt == 'u-cl029') | (expt == 'u-cl031')):
                l_gswp3 = 1
            if(l_gswp3 == 1):
                fn_expt = '/GSWP3-W5E5_OBSCLIM/isimip3a_notriffid_gswp3-w5e5_obsclim_historical.ilamb.'
                fn_expt2 = '/GSWP3-W5E5_OBSCLIM/isimip3a_notriffid_gswp3-w5e5_obsclim_historical.gen_mon_gb.'
                output_type1 = 'GSWP3_'
#                yre=2019
#            yre=2019 # always set to <= 2019 as GSWP doesnt go further?
# Start and end years:
#        yre = 2014 # GSWP3
#        yrb = 1979 # WFDEI
#        yre = 2015 # WFDEI, CRU
            yrb = 1990 # temporary
            yre = 1999 # temporary

            ny = yre-yrb+1
            cyr = str(yrb)

            datem = np.zeros([ny*nm],dtype = np.float32)
            for iy in range(ny):
                for im in range(nm):
                    datem[iy*nm+im] = yrb+iy+(sum(mon_len[0:im])+0.5*mon_len[im])/sum(mon_len[:])
                                                                     
#------------------------------------------------------
# Read in data for land and river grids:
# May not need to do this for each run 
            dir = '/hpc/data/d01/hadcam/jules_output/ALL_u-'+expt+'_isimip_0p5deg_origsoil_dailytrif/HADGEM2-ES/'  # '/hpc/data/d05/hadng/jules_output/'+expt+'/'
            fn1b = fn_expt+str(yrb)+'.nc'
            print(dir+fn1b)
            self.landIndex(fn1,dir+fn1b)
            
            self.dlatl = self.lat2d[1,0]-self.lat2d[0,0]
            self.dlonl = self.lon2d[0,1]-self.lon2d[0,0]
            
            print("self.dlonl,dlatl",self.dlonl,self.dlatl)
            print(self.lat2d[0,0],self.lon2d[0,0])

            lonl2d = np.array(self.lon2d[0,:]).flatten()
            latl2d = np.array(self.lat2d[:,0]).flatten()
            dlonl = lonl2d[1]-lonl2d[0]
            dlatl = latl2d[1]-latl2d[0]

#--------------------
# Read in river ancil (needed for upstream area):
# NB if the land grid changes this could cause the following to crash:
            if(ie == 0):
                file_river_ancil = '/hpc/data/d00/hadea/isimip3a/jules_ancils/rivers.latlon_fixed.nc'
                self.readFldFixed2d(file_river_ancil, mdi, "mystery1")
                aream = np.array(self.fld2d[:,:])    # Andy Wiltshire says this is upstream area
                aream = np.ma.masked_array(aream, aream < 0 )
                self.readFldFixed2d(file_river_ancil, mdi, "rivseq")
                rivseqm = np.array(self.fld2d[:,:])    # Andy Wiltshire says this is river order
                rivseqm = np.ma.masked_array(rivseqm, rivseqm < 0 )
                lonr2d = dc(lonl2d)
                latr2d = -dc(latl2d)
                dlonr = lonr2d[1]-lonr2d[0]
                dlatr = latr2d[1]-latr2d[0]

# Reverse so it's the same at the output
                aream = aream[::-1,:]
                rivseqm = rivseqm[::-1,:]
                latr2d = latr2d[::-1]
                dlatr = -dlatr

# Check that land and river grids are the same:
# Note that if are different or land grid changes with "ie" code will crash
# Plotting code not written to allow for this as yet.
            print('***CHECKING river and land model grids are the same***')
            if((np.abs(np.sum(latr2d-latl2d)) > 0.0) | (np.abs(np.sum(lonr2d-lonl2d)) > 0.0)):
                print('river and land model grids are different => stopping')
                print('np.abs(np.sum(latr2d-latl2d))',np.abs(np.sum(latr2d-latl2d)))
                print('np.abs(np.sum(lonr2d-lonl2d))',np.abs(np.sum(lonr2d-lonl2d)))
                sys.exit() 

#--------------------       
#-------------------------------------
# Get area of all grid boxes:
            self.getAreaWgt()

#
#-------------------------------------
# Reading in modelling river flow
            if(ie == 0):
                rflowS2_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)  
                rflowS2mm_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)  
                rflowS2m0_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64) 
                rflowS2mp_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)
                rflowS20m_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)
                rflowS20p_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)
                rflowS2pm_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)
                rflowS2p0_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)
                rflowS2pp_store = np.zeros([nex,ny*nm,nsoo],dtype = np.float64)
                rflowS2pm_m3persec_rescale = np.zeros([nex,ny*nm],dtype = np.float64)

            for iy in range(ny):
                yr=yrb+iy
                cyr=str(yr)
                fnS2 = fn_expt2+cyr+".nc" # CO2 and LUC varying
                
                print("reading year iy",iy)
                print('dir + fnS2',dir + fnS2)
                for im in range(nm):
                    
                    fld = "rflow"
                    self.readFld(dir + fnS2, mdi, im, fld)
                    rflowS2 = np.array(self.fld2d[:,:])           
                    
                    # reset any nans to missing data:
                    if((iy == 0) & (im == 0) & np.any(np.isnan(rflowS2))):
                        print("rflowS2 has missing data on the following number of points:" \
                              ,np.sum(np.isnan(rflowS2)))
                        rflowS2[np.isnan(rflowS2)] = mdi

# now extract data at required lats and lons:
                    for isi in range(nsoo):
                        if(dlonl > 0):
                            ii = ((lonl2d-0.5*dlonl <= lono[isi]) & (lonl2d+0.5*dlonl >= lono[isi]))
                        else:
                            ii = ((lonl2d-0.5*dlonl >= lono[isi]) & (lonl2d+0.5*dlonl <= lono[isi]))
                        
                        if(dlatl > 0):
                            jj = ((latl2d-0.5*dlatl <= lato[isi]) & (latl2d+0.5*dlatl >= lato[isi]))
                        else:
                            jj = ((latl2d-0.5*dlatl >= lato[isi]) & (latl2d+0.5*dlatl <= lato[isi]))
                            
# Bug correction only relevant for JULES vn4.7(vn4.8?) and before:
                        if(l_bug == 1):
                            if(dlonl > 0):
                                ii = ((lonl2d-0.5*dlonl <= lono[isi]-dlonl) & (lonl2d+0.5*dlonl >= lono[isi]-dlonl))
                            else:
                                ii = ((lonl2d-0.5*dlonl >= lono[isi]-dlonl) & (lonl2d+0.5*dlonl <= lono[isi]-dlonl))
                            if(dlatl > 0):
                                jj = ((latl2d-0.5*dlatl <= lato[isi]-dlatl) & (latl2d+0.5*dlatl >= lato[isi]-dlatl))
                            else:
                                jj = ((latl2d-0.5*dlatl >= lato[isi]-dlatl) & (latl2d+0.5*dlatl <= lato[isi]-dlatl))

                        jj = dc(jj.argmax())
                        ii = dc(ii.argmax())

# Stored river flow in adjacent grid boxes to that with obs station in it, in case needed for manual correction:
                        jjm= jj-1
                        jjp= jj+1
                        iim= ii-1
                        iip= ii+1
                        rflowS2_store[ie,iy*nm+im,isi] = rflowS2[jj,ii]  # S2 = experiment  # rflowS2 - if the model was perfect
                        rflowS2mm_store[ie,iy*nm+im,isi] = rflowS2[jjm,iim]  # mm = rflowS2 x-1 and y-1
                        rflowS2m0_store[ie,iy*nm+im,isi] = rflowS2[jjm,ii]
                        rflowS2mp_store[ie,iy*nm+im,isi] = rflowS2[jjm,iip]
                        rflowS20m_store[ie,iy*nm+im,isi] = rflowS2[jj,iim]
                        rflowS20p_store[ie,iy*nm+im,isi] = rflowS2[jj,iip]
                        rflowS2pm_store[ie,iy*nm+im,isi] = rflowS2[jjp,iim]
                        rflowS2p0_store[ie,iy*nm+im,isi] = rflowS2[jjp,ii]
                        rflowS2pp_store[ie,iy*nm+im,isi] = rflowS2[jjp,iip]


# Now plot data:
        file_pdf = 'multipage_pdf_'+expt+'_dai2014_global_vn1p1.pdf'
        if os.path.exists(file_pdf):
            os.remove(file_pdf)   # remove old file

        with PdfPages(file_pdf) as pdf:
            diro = '/project/jchmr/hadng/ancil/danda/'

#            nsoplot = nso
#            nsoplot = 20 # temporary
            nsoplot = 10 # # WARNING only manually corrected for the first 10 rivers

            for isi in range(nsoplot):

                if(dlonl > 0):
                    ii = ((lonl2d-0.5*dlonl <= lono[isi]) & (lonl2d+0.5*dlonl >= lono[isi]))
                else:
                    ii = ((lonl2d-0.5*dlonl >= lono[isi]) & (lonl2d+0.5*dlonl <= lono[isi]))
                if(dlatl > 0):
                    jj = ((latl2d-0.5*dlatl <= lato[isi]) & (latl2d+0.5*dlatl >= lato[isi]))
                else:
                    jj = ((latl2d-0.5*dlatl >= lato[isi]) & (latl2d+0.5*dlatl <= lato[isi]))
# Only relevant for JULES vn4.7(vn4.8?) and before:
                if(l_bug == 1):
                    if(dlonl > 0):
                        ii = ((lonl2d-0.5*dlonl <= lono[isi]-dlonl) & (lonl2d+0.5*dlonl >= lono[isi]-dlonl))
                    else:
                        ii = ((lonl2d-0.5*dlonl >= lono[isi]-dlonl) & (lonl2d+0.5*dlonl <= lono[isi]-dlonl))
                    if(dlatl > 0):
                        jj = ((latl2d-0.5*dlatl <= lato[isi]-dlatl) & (latl2d+0.5*dlatl >= lato[isi]-dlatl))
                    else:
                        jj = ((latl2d-0.5*dlatl >= lato[isi]-dlatl) & (latl2d+0.5*dlatl <= lato[isi]-dlatl))
                    
                jj = dc(jj.argmax())
                ii = dc(ii.argmax())

                print('jj,ii',jj,ii)

# rescale river flow from kg/m2/s -> kg/s -> m^3/s:
                print(self.river_list[isi] \
                          +" "+str(lato[isi])+"N,"+str(lono[isi])+"E")
                print('isi,lono[isi]',isi,lono[isi],'lato[isi]',lato[isi])
                print('latl2d[jj],lonl2d[ii]',latl2d[jj],lonl2d[ii])
                print('self.area2d[jj,ii]',self.area2d[jj,ii])
                print('rflowS2_store[0:3,isi]',isi,(rflowS2_store[ie,0:3,isi]))
                
#---------------------------------------------------------
# Investigate the river flow at grid boxes adjacent to the
# one which contains the gauge station:
                jjm = jj-1
                jjp = jj+1
                iim = ii-1
                iip = ii+1
                rflowS2_m3persec = (rflowS2_store[:,:,isi]*(self.area2d[jj,ii]/1000.))
                rflowS2mm_m3persec = (rflowS2mm_store[:,:,isi]*(self.area2d[jjm,iim]/1000.))
                rflowS2m0_m3persec = (rflowS2m0_store[:,:,isi]*(self.area2d[jjm,ii]/1000.))
                rflowS2mp_m3persec = (rflowS2mp_store[:,:,isi]*(self.area2d[jjm,iip]/1000.))
                rflowS20m_m3persec = (rflowS20m_store[:,:,isi]*(self.area2d[jj,iim]/1000.))
                rflowS20p_m3persec = (rflowS20p_store[:,:,isi]*(self.area2d[jj,iip]/1000.))
                rflowS2pm_m3persec = (rflowS2pm_store[:,:,isi]*(self.area2d[jjp,iim]/1000.))
                rflowS2p0_m3persec = (rflowS2p0_store[:,:,isi]*(self.area2d[jjp,ii]/1000.))
                rflowS2pp_m3persec = (rflowS2pp_store[:,:,isi]*(self.area2d[jjp,iip]/1000.))
            
                dareamo = np.abs(aream[jj,ii]-areao[isi])
                dareamo_mm = np.abs(aream[jjm,iim]-areao[isi])
                dareamo_m0 = np.abs(aream[jjm,ii]-areao[isi])
                dareamo_mp = np.abs(aream[jjm,iip]-areao[isi])
                dareamo_0m = np.abs(aream[jj,iim]-areao[isi])
                dareamo_0p = np.abs(aream[jj,iip]-areao[isi])
                dareamo_pm = np.abs(aream[jjp,iim]-areao[isi])
                dareamo_p0 = np.abs(aream[jjp,ii]-areao[isi])
                dareamo_pp = np.abs(aream[jjp,iip]-areao[isi])
                iiuse = dc(ii)
                jjuse = dc(jj)
                dareamo_use = dc(dareamo)
                rflowS2use_m3persec = dc(rflowS2_m3persec)

#-------------------------------------------------------------
# Rivers that need manual/automated grid box corrections:
# In the case of automated correction chose adjacentent grid box which has the upstream area which is closest to
# that in the Dai et al file:
                lcorrect = 0
                rivers2correct=['Orinoco','Mississippi','Yenisey','St Lawrence','Amur','Mackenzie','Xijiang']
                nrivers2correct =len(rivers2correct)
                for ir in range(nrivers2correct):
                    if(self.river_list[isi].rstrip() == rivers2correct[ir]):
                        lcorrect = 1
                        print('lcorrect',lcorrect,self.river_list[isi].rstrip())

                if(lcorrect == 1):
                    if(dareamo_mm < dareamo_use):
                        dareamo_use = dc(dareamo_mm) 
                        jjuse = dc(jjm)
                        iiuse = dc(iim)
                        rflowS2use_m3persec = dc(rflowS2mm_m3persec)
                    if(dareamo_m0 < dareamo_use):
                        dareamo_use = dc(dareamo_m0) 
                        jjuse = dc(jjm)
                        iiuse = dc(ii)
                        rflowS2use_m3persec = dc(rflowS2m0_m3persec)
                    if(dareamo_mp < dareamo_use):
                        dareamo_use = dc(dareamo_pm) 
                        jjuse = dc(jjm)
                        iiuse = dc(iip)
                        rflowS2use_m3persec = dc(rflowS2mp_m3persec)
                    if(dareamo_0m < dareamo_use):
                        dareamo_use = dc(dareamo_0m) 
                        jjuse = dc(jj)
                        iiuse = dc(iim)
                        rflowS2use_m3persec = dc(rflowS20m_m3persec)
                    if(dareamo_0p < dareamo_use):
                        dareamo_use = dc(dareamo_0p) 
                        jjuse = dc(jj)
                        iiuse = dc(iip)
                        rflowS2use_m3persec = dc(rflowS20p_m3persec)
                    if(dareamo_pm < dareamo_use):
                        dareamo_use = dc(dareamo_pm) 
                        jjuse = dc(jjp)
                        iiuse = dc(iim)
                        rflowS2use_m3persec = dc(rflowS2pm_m3persec)
                    if(dareamo_p0 < dareamo_use):
                        dareamo_use = dc(dareamo_p0) 
                        jjuse = dc(jjm)
                        iiuse = dc(iim)
                        rflowS2use_m3persec = dc(rflowS2p0_m3persec)
                    if(dareamo_pp < dareamo_use):
                        dareamo_use = dc(dareamo_pp) 
                        jjuse = dc(jjp)
                        iiuse = dc(iip)
                        rflowS2use_m3persec = dc(rflowS2pp_m3persec)

# Manually overwrite grid box chosen if necessary after looking at figures in pdf: 
                    if((self.river_list[isi].rstrip() == 'Orinoco') | \
                       (self.river_list[isi].rstrip() == 'Yenisey')  | \
                       (self.river_list[isi].rstrip() == 'Xijiang')):
                        dareamo_use = dc(dareamo_0p) 
                        jjuse = dc(jj)
                        iiuse = dc(iip)
                        rflowS2use_m3persec = dc(rflowS20p_m3persec)
                    if(self.river_list[isi].rstrip() == 'Mississippi'):
                        dareamo_use = dc(dareamo_0m) 
                        jjuse = dc(jj)
                        iiuse = dc(iim)
                        rflowS2use_m3persec = dc(rflowS20m_m3persec)

                    print('aream,areao,darea,rivseq',jj,ii,aream[jj,ii],areao[isi],dareamo,rivseqm[jj,ii])
                    print('aream[jj,ii]',jj,ii,aream[jj,ii],areao[isi],dareamo,rivseqm[jj,ii] )
                    print('areamm[jjm,iim]',jjm,ii,aream[jjm,iim],areao[isi],dareamo_mm,rivseqm[jjm,iim])
                    print('aream0[jjm,ii]',jjm,ii,aream[jjm,ii],areao[isi],dareamo_m0,rivseqm[jjm,ii])
                    print('areamp[jjm,iip]',jjm,ii,aream[jjm,iip],areao[isi],dareamo_mp,rivseqm[jjm,iip])
                    print('aream0m[jj,iim]',jj,ii,aream[jj,iim],areao[isi],dareamo_0m,rivseqm[jj,iim])
                    print('aream0p[jj,iip]',jj,ii,aream[jj,iip],areao[isi],dareamo_0p,rivseqm[jj,iip])
                    print('areampm[jjp,iim]',jjp,iim,aream[jjp,iim],areao[isi],dareamo_pm,rivseqm[jjp,iim])
                    print('areamp0[jjp,ii]',jjp,ii,aream[jjp,ii],areao[isi],dareamo_p0,rivseqm[jjp,ii])
                    print('areampp[jjp,iip]',jjp,iip,aream[jjp,iip],areao[isi],dareamo_pp,rivseqm[jjp,iip])

#-------------------------------------------------------------
# Rescale river flow to allow for the fact the river ancil
# has a different upscale area to observation:
                if(aream[jj,ii] > 0):
                    rflowS2_m3persec_rescale = np.array(rflowS2_m3persec/aream[jj,ii]*areao[isi])
                    rflowS2use_m3persec_rescale = np.array(rflowS2use_m3persec/aream[jjuse,iiuse]*areao[isi])
                else:
                    rflowS2_m3persec_rescale[:] = -999.
                    
# Store non missing river flow data in readiness for plotting:
                vals = dateo[isi,:] > 0
                yearoo = dc(yearo[isi,vals])
                monoo = dc(mono[isi,vals])

                rflowoo = dc(rflowo[isi,vals])
                rflowoot = dc(rflowS2_m3persec_rescale[0,:].flatten())
                rflowoot[:] = -999.

                it = -1
                for iy in range(ny):
                    for im in range(nm):
                        it = it+1
                        vals_it = (monoo == np.float32(im+1)) & (yearoo == np.float32(yrb+iy))
                        
                        if(vals_it.sum() > 0):
                            rflowoo_avg = np.mean(rflowoo[vals_it])
                        else:
                            rflowoo_avg = -999.
                            
                        rflowoot[it] = rflowoo_avg 


                rflowoot_tmp = np.reshape(rflowoot,(ny,nm))
                rflowoot_mmm = np.zeros(nm)
                rflowoot_mmm[:] = -999.
                rflowS2use_m3persec_tmp = np.reshape(rflowS2use_m3persec,(nex,ny,nm))
                rflowS2use_m3persec_mmm = np.zeros([nex,nm])
                rflowS2use_m3persec_rescale_tmp = np.reshape(rflowS2use_m3persec_rescale,(nex,ny,nm))
                rflowS2use_m3persec_rescale_mmm = np.zeros([nex,nm])
                rflowS2_m3persec_tmp = np.reshape(rflowS2_m3persec,(nex,ny,nm))
                rflowS2_m3persec_mmm = np.zeros([nex,nm])
                rflowS2_m3persec_rescale_tmp = np.reshape(rflowS2_m3persec_rescale,(nex,ny,nm))
                rflowS2_m3persec_rescale_mmm = np.zeros([nex,nm])

                for im in range(nm):
                    tmp = rflowoot_tmp[:,im].flatten()
                    vals_mm = (tmp > -999.)
                    rflowoot_mmm[im] = np.mean(tmp[vals_mm])
                    if(np.sum(len(vals_mm)) < ny):
                        print('vals_mm',vals_mm)
                        print(rflowoot_tmp[:,im],rflowoot_mmm[im])
                    for ie in range(nex):
                        tmp = rflowS2use_m3persec_tmp[ie,:,im].flatten()
                        rflowS2use_m3persec_mmm[ie,im] = np.mean(tmp)
                        tmp = rflowS2use_m3persec_rescale_tmp[ie,:,im].flatten()
                        rflowS2use_m3persec_rescale_mmm[ie,im] = np.mean(tmp)
                        tmp = rflowS2_m3persec_tmp[ie,:,im].flatten()
                        rflowS2_m3persec_mmm[ie,im] = np.mean(tmp)
                        tmp = rflowS2_m3persec_rescale_tmp[ie,:,im].flatten()
                        rflowS2_m3persec_rescale_mmm[ie,im] = np.mean(tmp)


# Plotting:
#                iplot = (isi+1) % nsplots
                iplot = ((isi+1) % nrows) # to allow river sequence plot

                print('isi+1,iplot,nrows,ncols,((isi+1) % nrows)')
                print(isi+1,iplot,nrows,ncols,((isi+1) % nrows))


                if(iplot == 0):
#                    iplot = nsplots
                    iplot = nrows # to allow river sequence plot

                if(iplot == 1):
                    plt.figure(figsize = (7, 10)) # portrait


# Start of plots:
#                plt.subplot(nrows,ncols,iplot)
                plt.subplot(nrows,ncols,iplot*2-1) # to allow river sequence plot
                plt.subplots_adjust(wspace=0.75, hspace=0.5)
                plt.title(self.river_list[isi].rstrip()+ "\n"+stationor_list[isi] \
                          +" "+str(lato[isi])+"N,"+str(lono[isi])+"E", fontsize=8)

# Now plot monthly means: 
#                plt.xlim([yrb,yre+1])
#                
#                vals = rflowoot > -999.                
#                plt.plot(datem[vals],rflowoot[vals],'black')
#                plt.ylim(0,1.05*np.max([np.max(rflowS2use_m3persec) \
#                                        ,np.max(rflowS2use_m3persec_rescale),np.max(rflowoot[vals])]))
#                for ie in range(nex):
##                    plt.plot(datem[:],rflowS2_m3persec_rescale[ie,:],'blue',linewidth=lthin,linestyle=lstyle[ie])
##                    plt.plot(datem[:],rflowS2_m3persec[ie,:],'red',linewidth=lthin,linestyle=lstyle[ie])
## The riverflow for the corrected grid boxes:
#                    plt.plot(datem[:],rflowS2use_m3persec_rescale[ie,:],'blue',linestyle=lstyle[ie])
#                    plt.plot(datem[:],rflowS2use_m3persec[ie,:],color='red',linestyle=lstyle[ie])


# Now plot multi monthly means:
                vals = rflowoot_mmm > -999.                
                plt.plot(month,rflowoot_mmm[vals],'black')
                plt.xlabel('month')
                plt.ylim(0,1.05*np.max([np.max(rflowS2use_m3persec_mmm) \
                                        ,np.max(rflowS2use_m3persec_rescale_mmm),np.max(rflowoot_mmm)]))
                for ie in range(nex):
#                    plt.plot(month,rflowS2_m3persec_rescale_mmm[ie,:],'blue',linewidth=lthin,linestyle=lstyle[ie])
#                    plt.plot(month,rflowS2_m3persec_mmm[ie,:],'red',linewidth=lthin,linestyle=lstyle[ie])
# The riverflow for the corrected grid boxes:
                    plt.plot(month,rflowS2use_m3persec_rescale_mmm[ie,:],'blue',linestyle=lstyle[ie])
                    plt.plot(month,rflowS2use_m3persec_mmm[ie,:],color='red',linestyle=lstyle[ie])
                
#------------------
# Image river sequence plot to check ancil river against gauge station:
# Plot around the central point:
                iimin = iim-1
                iimax = iip+1
                jjmin = jjm-1
                jjmax = jjp+1
                lontl=lonr2d[iimin]-0.5*dlonr
                lontr=lonr2d[iimax]+0.5*dlonr
                latbl=latr2d[jjmin]-0.5*dlatr
                latbr=latr2d[jjmax]+0.5*dlatr
                print('lontl,lontr,latbl,latbr',lontl,lontr,latbl,latbr)

                plt.subplot(nrows,ncols,2*iplot)
                plt.imshow(rivseqm[jjmin:jjmax+1,iimin:iimax+1],extent=[lontl,lontr,latbl,latbr])
                plt.plot([lono[isi],lono[isi]],[lato[isi],lato[isi]],marker='X',color='white')
                plt.title("river sequence "+stationor_list[isi]+"\n"+'jjuse-jj,iiuse-ii:' \
                          +str(jjuse-jj)+','+str(iiuse-ii),fontsize=8)
                plt.colorbar()

                plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.7)

#                if(iplot == nsplots or isi == nsoplot-1):
                if(iplot == nrows or isi == nsoplot-1): # to allow river sequence plot
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                    

if __name__ == "__main__":
    t = test()
    t.doStuff()

##
##
