##### ffmo 111722
#### making for Gino
#### copying parts from AP_Cloud_LongRecord_2.R which was processing before Schwartz et al 2014 paper 


################################################################################# coastal airports with long records provided to me by Sam 
FilePath <-list.files("../FromSam/airportcld_longerRecords/", pattern = "*.hourly.txt")
APname <- substr(FilePath,1,4)


length(timePSTs);153*24*63 ##### 153 days in MJJAS, 24 hrs/day, 63 yrs from 1950 to 2012 = 231,336 hours 
APs <- array(NA,dim=c(length(timePSTs),2,length(APname)),dimnames=list(timePST=as.character(timePST),metric=c("cld.frac","cld.base.m"),airport=APname))
cf <- count.fields("../FromSam/airportcld_longerRecords/stationdata.dat")
stati <- read.table(file="../FromSam/airportcld_longerRecords/stationdata_RESforR.txt")
########### match the coastal long record stations I am using to the station info here... look at elevation range e.g. 
miStat <-match(dimnames(APs)[[3]],stati[,1])
cbind(stati[miStat,1],dimnames(APs)[[3]]) #### matched up
stati[miStat,c(1,2,7)] ##### this is elevation in meters
range(stati[miStat,7])
##### need to just add KAST which didn't have data at first
f <- 3
#for (f in 1:length(FilePath)){
    AP <-  as.matrix(read.table(paste("../FromSam/airportcld_longerRecords/",FilePath[f],sep="")))
    colnames(AP) <- c("year", "month", "day", "hour", "cld.frac", "cld.base.ft")
    AP[which(AP==-999|AP==-999.9)] <- NA ## these are missing to NA  update 
    AP[which(AP==72200)] <- NaN ### not really missing per say, just when cloud fraction low, given this for base height
    print("time to chron")
    timePSTall <- as.chron(strptime(paste(AP[,"year"],AP[,"month"],AP[,"day"],AP[,"hour"],sep="/"),"%Y/%m/%d/%H")) - 8/24 ## change from UTC to PST
    ##### Note since the slow part is "chroning" I could match times not with chron but in some other numeric format 
    monthsi <- which(as.numeric(months(timePSTall))>=5&as.numeric(months(timePSTall))<=9) ### MJJAS=  5 to 9 months 
    #monthsi <- which(as.numeric(months(timePSTall))>=5&as.numeric(months(timePSTall))<=9) ### MJJAS=  4 to 10 months  (do this if need for 5 to 9 seasonal cycle calc)
    timePST <- timePSTall[monthsi]
    APs1 <- AP[monthsi,c("cld.frac","cld.base.ft")]### Airport Summer, goes with timePST
############### match times ... that may have some missing, with the complete hourly smooth time series 
    mi<- c(1:length(timePSTs))[as.character(timePSTs)%in%as.character(timePST)] ### remember use as.character with chron times, to avoid floating numbers
    range(timePSTs[mi]-timePST)###### check should be zeros 
##### now put in correct spot, others (missing time steps) will be NAs     
    APs[mi,,f] <- APs1 ### will get an error if wrong size
    APs[mi,2,f] <- APs[mi,2,f]*0.3048 + (stati[miStat,7][f])#### change to meters and change to asl by adding the station elevation provided by Sam 
    print(f)
}
APsKSAT <- APs[,,f]
#### save file
save(APs,file="./outputs/APs_cldhourly_NaN_MtoS_1950to2012_masl_addNorth_080713.rda") 


########## SUBSET to just hours and airports I want
### just hours and airport want 
APsW <- APs[hrsWi,,c(2,4,5,9,11,12,13,15,17,20,21,22)] 

MonLCf <- array(NA,dim=c(length(MonthYr),dim(APsW)[[3]]),dimnames=list(MonthYr=MonthYr,AP=dimnames(APsW)[[3]]))#### Monthly low cloud frequency 
dim(MonLCf)
c <- 0 
for(yi in 1:length(yrs)){PNonMissMonHr
    for(mi in c("May","Jun","Jul","Aug","Sep")){
        c <- c +1
        yis <- which(years(timePSTW)==yrs[yi]&months(timePSTW)==mi)
        for (ai in 1:dim(APsW)[3]){

            nobs <- length(which(is.na(APsW[yis,"cld.base.m",ai])==FALSE&APsW[yis,"cld.frac",ai]>0.7)) + length(which(APsW[yis,"cld.frac",ai]<0.7)) 

            vlowcld <- length(which(APsW[yis,"cld.base.m",ai]<1000&APsW[yis,"cld.frac",ai]==1.00)) +
                length(which(APsW[yis,"cld.base.m",ai]<1000&APsW[yis,"cld.frac",ai]==0.75))

            MonLCf[c,ai] <- vlowcld/nobs*100
        }
    }
    print(c)
}
####################################### put NAs at the times off too low of valid data < 25% valid for any of the hours
##### ###### from AP_Cloud_LongRecord.R 
load(file="./outputs/PNonMissMonHr_APobs_North_080813.rda")
dimnames(PNonMissMonHr)[[2]]
PvalidN <- PNonMissMonHr
load(file="./outputs/PNonMissMonHr_APobs_071713.rda")
dimnames(PNonMissMonHr)[[2]]
dim(PNonMissMonHr)
miNorth <- match(dimnames(PvalidN)[[2]],dimnames(MonLCf)[[2]])
miMore <- match(dimnames(PNonMissMonHr)[[2]],dimnames(MonLCf)[[2]])
Pall <- array(NA,c(dim(MonLCf)[1],dim(MonLCf)[2],24),dimnames=list(monthyr=dimnames(MonLCf)[[1]],ap=dimnames(MonLCf)[[2]],hrs=dimnames(PNonMissMonHr)[[3]]))
dim(Pall)
Pall[,miNorth,] <- PvalidN
Pall[,c(1:3,4:6,8:9),] <- PNonMissMonHr[,c(1:3,5:9),] ### NA's are not allowed, KNSI out... 

###

################################################################## Ok now, what indicies should I set to NA because not enough valid data...
MonLCfog <- MonLCf
length(which(is.na(MonLCf)==TRUE)) #### 40 to start with ... they are all in PSIT because the numbers are actually zero.. 
length(which(is.na(MonLCfog[,10])==TRUE))  

for (ai in 1:dim(Pall)[2]){
    for (ti in 1:dim(Pall)[1]){
        if(any(Pall[ti,ai,c(8,11,14,17,20)]<25))  ### requirement at least 25% of the data valid for each time=step  !!!! RSC 11/17/22 shouldn't it be 7, 10, 13, and 16 PST
            MonLCf[ti,ai] <- NA ### then set to NA 
    }
    print(ai)    
}
length(which(is.na(MonLCf)==TRUE)) #### now 71... NAs ... i think cc() should be able to handle... let's see 
MonLCf[,"KSAN"] ### correct, put it at August 1996 ...
which(is.na(MonLCf[,"KAST"]==TRUE))
length(which(is.na(MonLCf[,"PSIT"]==TRUE)))
which(is.na(MonLCf[,"PYAK"]==TRUE)) ### should be no NAs , correct 

##############################
MonLCfl <- list(MonLCf=MonLCf,CenterTimePST = monthyrC)
save(MonLCfl,file="./outputs/MonLCfl_monthlyLowCloudFreq_AP12_080913.rda")
