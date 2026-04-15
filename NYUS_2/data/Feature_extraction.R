#NYUS.2 feature generation
##Please contact hw692@cornell.edu/hongrui.wang@inrae.fr for any additional questions.

#Due to the limited availability of hourly temperature data, this script aims to extract features from daily temperature data.
#Load libraries

require(dplyr)                                    
require(readr)  
require(tidyverse)
require(ggpubr)
require(wesanderson)
require(pracma)
require(data.table)
require(weathermetrics)
require(measurements)
require(naniar)
require(ggplot2)
require(ggpubr)
require(dormancyR)
require(chron)
require(ggrepel)
require(geosphere)
require(lubridate)
require(caret)
require(fruclimadapt)

args <- commandArgs(trailingOnly = TRUE)

input_file <- args[1]
latitude <- as.numeric(args[2])
cultivar <- args[3]

#Load cultivars
load('NYUS_2/data/Cultivars_NYUS_2_2.Rdata')
cultivars_to_choose_from = gsub('Cultivar.','',Cultivars)

#Load a demo raw temperature file
#The demo temperature file includes three columns: one date column, one max daily temperature column and one min daily temperature column
all_data = read_csv(input_file)
all_data$tmax = as.numeric(all_data$tmax)
all_data$tmin = as.numeric(all_data$tmin)

all_data$date = mdy(all_data$date)
colnames(all_data)
colnames(all_data)[2:3] = c('Tmax','Tmin')


#As Tmax and Tmin are is in Fahrenheit, the first step is to change the unit to Celsius
all_data$Tmax = fahrenheit.to.celsius(all_data$Tmax)
all_data$Tmin = fahrenheit.to.celsius(all_data$Tmin)




###Functions
step_model<-function(HourTemp,
                     df=data.frame(
                       lower=c(-1000,1.4,2.4,9.1,12.4,15.9,18),
                       upper=c(1.4,2.4,9.1,12.4,15.9,18,1000),
                       weight=c(0,0.5,1,0.5,0,-0.5,-1)),summ=TRUE)
{lower<-df$lower;upper<-df$upper;weight<-df$weight
if (summ==TRUE) return(cumsum(sapply(HourTemp,function(x) weight[which(x>lower&x<=upper)]))) else
  return(sapply(HourTemp,function(x) weight[which(x>lower&x<=upper)]))
}
#inverse_EWA is the function to compute REWMA temperatures
inverse_EWA = function(datalist,window=10){
  datalist_rev = rev(datalist)
  datalist_rev_EWA = movavg(datalist_rev,n = window,type = 'e')
  datalist_EWA = rev(datalist_rev_EWA)
  datalist_EWA = c(rep(NA,window), datalist_EWA)
  datalist_EWA = datalist_EWA[1:length(datalist)]
  return(datalist_EWA)
}

daylength<-function(latitude,JDay,notimes.as.na=FALSE)
{
  if(missing(latitude)) stop("'latitude' not specified")
  if(missing(JDay)) stop("'JDay' not specified")
  if (!isTRUE(all(is.numeric(JDay)))) stop("'JDay' contains non-numeric values")
  if(length(latitude)>1) stop("'latitude' has more than one element")
  if (!is.numeric(latitude)) stop("'latitude' is not numeric")
  if(latitude>90|latitude<(-90)) warning("'latitude' is usually between -90 and 90")

  Gamma<-2*pi/365*((JDay)-1)
  Delta<-180/pi*(0.006918-0.399912*cos(Gamma)+0.070257*sin(Gamma)-0.006758*cos(2*Gamma)+
                   0.000907*sin(2*(Gamma))-0.002697*cos(3*(Gamma))+0.00148*sin(3*(Gamma)))
  CosWo<-(sin(-0.8333/360*2*pi)-sin(latitude/360*2*pi)*
            sin(Delta/360*2*pi))/(cos(latitude/360*2*pi)*cos(Delta/360*2*pi))
  
  normal_days<-which(CosWo>=-1&CosWo<=1)
  Sunrise<-rep(-99,length(CosWo))
  Sunrise[normal_days]<-12-acos(CosWo[normal_days])/(15/360*2*pi)
  Sunset<-rep(-99,length(CosWo))
  Sunset[normal_days]<-12+acos(CosWo[normal_days])/(15/360*2*pi)
  Daylength<-Sunset-Sunrise
  Daylength[which(CosWo>1)]<-0
  Daylength[which(CosWo<(-1))]<-24
  Sunrise[which(Daylength==24)]<-99
  Sunset[which(Daylength==24)]<-99
  
  if(notimes.as.na)
   {Sunrise[which(Sunrise %in% c(-99,99))]<-NA
    Sunset[which(Sunset %in% c(-99,99))]<-NA}
  Sunset[which(is.na(JDay))]<-NA
  Sunrise[which(is.na(JDay))]<-NA
  Daylength[which(is.na(JDay))]<-NA
 
 return(list(Sunrise=Sunrise,Sunset=Sunset,Daylength=Daylength))  
}

make_hourly_temps <-function (latitude,year_file,keep_sunrise_sunset=FALSE)
    
  {
    
    if(missing(latitude)) stop("'latitude' not specified")
    if(length(latitude)>1) stop("'latitude' has more than one element")
    if(!is.numeric(latitude)) stop("'latitude' is not numeric")
    if(latitude>90|latitude<(-90)) warning("'latitude' is usually between -90 and 90")
    
    
    year_file<-year_file[which(!is.na(year_file$Tmin)&!is.na(year_file$Tmax)),]
    
    if(!"JDay" %in% colnames(year_file))
      year_file[,"JDay"]<-strptime(paste(year_file$Month,"/",year_file$Day,"/",year_file$Year,sep=""),"%m/%d/%Y")$yday+1
    
    preserve_columns<-colnames(year_file)
    
    Day_times<-daylength(latitude=latitude,
                         JDay=c(year_file$JDay[1]-1,
                                year_file$JDay,
                                year_file$JDay[nrow(year_file)]+1))
    Day_times$Sunrise[which(Day_times$Sunrise==99)]<-0
    Day_times$Sunrise[which(Day_times$Sunrise==-99)]<-12
    Day_times$Sunset[which(Day_times$Sunset==99)]<-24
    Day_times$Sunset[which(Day_times$Sunset==-99)]<-12
    
    year_file$Sunrise<-Day_times$Sunrise[2:(length(Day_times$Sunrise)-1)]
    year_file$Sunset<-Day_times$Sunset[2:(length(Day_times$Sunset)-1)]
    year_file$Daylength<-Day_times$Daylength[2:(length(Day_times$Daylength)-1)]
    year_file$prev_Sunset<-Day_times$Sunset[1:(length(Day_times$Sunset)-2)]
    year_file$next_Sunrise<-Day_times$Sunrise[3:length(Day_times$Sunrise)]
    year_file$prev_max<-year_file$Tmax[c(NA,1:(nrow(year_file)-1))]
    year_file$next_min<-year_file$Tmin[c(2:nrow(year_file),NA)]
    year_file$prev_min<-year_file$Tmin[c(NA,1:(nrow(year_file)-1))]
    year_file$Tsunset<-year_file$Tmin+(year_file$Tmax-year_file$Tmin)*
      sin((pi*(year_file$Sunset-year_file$Sunrise)/(year_file$Daylength+4)))
    year_file$prev_Tsunset<-year_file$prev_min+(year_file$prev_max-year_file$prev_min)*
      sin((pi*(year_file$Daylength)/(year_file$Daylength+4)))
    colnum<-ncol(year_file)+1
    
    hourcol<-c(colnum:(colnum+23))
    
    
    
    
    for (hour in 0:23)
    {
      hourcount<-hour+1
      
      #if(length(which(year_file$Daylength==-99))>0)
      #{
      no_riseset<-which(year_file$Daylength %in% c(0,24,-99))
      year_file[no_riseset,colnum+hour]<-((year_file$Tmax+year_file$Tmin)/2)[no_riseset]
      #}
      
      c_morn<-which(hour<=year_file$Sunrise)
      if(1 %in% c_morn)
        if(!length(c_morn)==1) 
          c_morn<-c_morn[2:length(c_morn)]
      else c_morn<-c() #can't compute temperatures before sunrise for day 1
      c_day<-which(hour>year_file$Sunrise&hour<=year_file$Sunset)
      c_eve<-which(hour>=year_file$Sunset)
      if(nrow(year_file) %in% c_eve) c_eve<-c_eve[1:(length(c_eve)-1)] #can't compute temperatures after sunset for last day
      
      
      year_file[c_morn,colnum+hour]<-
        year_file$prev_Tsunset[c_morn]-  #prev temp at sunset
        ((year_file$prev_Tsunset[c_morn]-year_file$Tmin[c_morn])/
           log(pmax(1,24-(year_file$prev_Sunset[c_morn]-year_file$Sunrise[c_morn])))*
           log(hour+24-year_file$prev_Sunset[c_morn]+1))
      
      year_file[c_day,colnum+hour]<-
        year_file$Tmin[c_day]+
        (year_file$Tmax[c_day]-year_file$Tmin[c_day])*
        sin((pi*(hour-year_file$Sunrise[c_day])/
               (year_file$Daylength[c_day]+4)))
      
      year_file[c_eve,colnum+hour]<-
        year_file$Tsunset[c_eve]- #temp at sunset
        ((year_file$Tsunset[c_eve]-year_file$next_min[c_eve])/
           log(24-(year_file$Sunset[c_eve]-year_file$next_Sunrise[c_eve])+1)*
           log(hour-year_file$Sunset[c_eve]+1))
      
    }
    colnames(year_file)[(ncol(year_file)-23):(ncol(year_file))]<-c(paste("Hour_",0:23,sep=""))
    if (!keep_sunrise_sunset)
      year_file<-year_file[,c(preserve_columns,paste("Hour_",0:23,sep=""))]
    if (keep_sunrise_sunset)
      year_file<-year_file[,c(preserve_columns,"Sunrise","Sunset","Daylength",paste("Hour_",0:23,sep=""))]
    year_file[1,(ncol(year_file)-23):(ncol(year_file))][which(is.na(year_file[1,(ncol(year_file)-23):(ncol(year_file))]))]<-year_file[1,"Tmin"]
    year_file[nrow(year_file),(ncol(year_file)-23):(ncol(year_file))][which(is.na(year_file[nrow(year_file),(ncol(year_file)-23):(ncol(year_file))]))]<-year_file[nrow(year_file),"Tmin"]
    
    return(year_file)
  }



stack_hourly_temps <-function(weather=NULL,latitude=50,hour_file=NULL,keep_sunrise_sunset=FALSE)
   {
  if(length(latitude)>1) stop("'latitude' has more than one element")
  if(!is.numeric(latitude)) stop("'latitude' is not numeric")
  if(latitude>90|latitude<(-90)) warning("'latitude' is usually between -90 and 90")
  
  if(is.null(weather)&(!is.null(hour_file))) weather<-hour_file
     #this part is for the case where users specify a named variable 'hour_file'
     #This is necessary for backward compatibility
    if((length(names(weather))==2) & ("weather" %in% names(weather)) & ("QC" %in% names(weather))) 
      {if (!keep_sunrise_sunset) THourly<-make_hourly_temps(latitude,weather$weather)
       if (keep_sunrise_sunset) THourly<-make_hourly_temps(latitude,weather$weather,
                                                           keep_sunrise_sunset=TRUE)
      QC<-weather$QC}
    if((length(names(weather))>24) & ("Hour_23" %in% names(weather)))
      {THourly<-weather
      QC<-NA}
   if((length(names(weather))>2) & ("Tmax" %in% names(weather)))
      {if (!keep_sunrise_sunset)
        THourly<-make_hourly_temps(latitude,weather)
       if (keep_sunrise_sunset)
         THourly<-make_hourly_temps(latitude,weather,keep_sunrise_sunset=TRUE)
      QC<-NA}   
    
    if(is.null(weather)&is.null(hour_file)) {tt=NA;QC=NA} else
      {preserve_columns<-colnames(THourly)[1:(ncol(THourly)-24)]
      THourly[,"index"]<-row(THourly)[,1]
      ss<-stack(THourly,select=c(paste("Hour_",0:23,sep="")))
      colnames(ss)<-c("Temp","Hour")
      ss$Hour<-as.numeric(sapply(strsplit(as.character(ss$Hour),"_"),"[",2))
      for (pc in preserve_columns)
      {ss[,pc]<-rep(THourly[,pc],24)}
      tt<-ss[with(ss, order(Year, JDay,Hour)), ]
         tt<-tt[c(preserve_columns,"Hour","Temp")]
      }
    return(list(hourtemps=tt,QC=QC))  
  }

chilling_units <- function(HourTemp, summ = TRUE){

  #Giving a value of zero to all days for then changing according to each situation.
  #This is 0 for hours in which Temp < -4.66 or Temp > 16 C
  
    chilling_weights <- rep(0, length(HourTemp))
  
  #Selecting the hours which fit the condition: -4.66 < Temp < 16
    
    relevant_hours <- which(HourTemp >= -4.66 & HourTemp <= 16)
  
    #Computing chill for such condition
    
      chilling_weights[relevant_hours] <- 3.13 * (((HourTemp[relevant_hours] + 4.66) / 10.93) ^ 2.10) * 
        exp(1)^ - ((HourTemp[relevant_hours] + 4.66) / 10.93) ^ 3.10
  
  #Correction factor for those chill values above 1
      
    chilling_weights[chilling_weights > 1] <- 1
  
  #End of the function
    
    if (summ == TRUE)
    return(cumsum(chilling_weights)) else return(chilling_weights)
}

modified_utah_model <- function(HourTemp, summ = TRUE){
  
  #Giving a value of zero to the whole record
  
    chilling_weights <- rep(0,length(HourTemp))
  
  #Selecting the hours which fit the condition for chill accumulation (0 < Temp <= 21)
  
    relevant_hours <- which(HourTemp > 0 & HourTemp <= 21)
  
    #Calculating chill values for those hours
    
      chilling_weights[relevant_hours] <- sin((2 *pi * HourTemp[relevant_hours]) / 28)
  
  #For hours above 21 C chill accumulation is reduced in 1 hour
      
    chilling_weights[which(HourTemp > 21)] <- -1
    
  #End of the function
  
    if (summ == TRUE)
      return(cumsum(chilling_weights)) else return(chilling_weights)}

north_carolina_model <- function(HourTemp, summ = TRUE){
  return(step_model(HourTemp,
                            df = data.frame(lower = c(-1000, -1.1, 1.6, 7.2, 13.0, 16.5, 19.0, 20.7, 22.1, 23.3),
                                            upper = c(-1.1, 1.6, 7.2, 13.0, 16.5, 19.0, 20.7, 22.1, 23.3, 1000),
                                            weight = c(0, 0.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2)), summ = summ))}
#Weather_feature_generation is the function to compute all the features need for NYUS.2
#This function requires three inputs:
#First, a df similar with 'all_data' that have a 'date' column (properly formatted for date datatype) and two numeric columns ('Tmin' and 'Tmax') containing daily maximum and minimum temperatures
#Second, the latitude of the site of the temperature data
#Third, your cultivar of interest for the prediction of freezing tolerance. Please copy and paste a cultivar from 'cultivars_to_choose_from'
Dynamic_Model <- function(HourTemp,
                          summ = TRUE,
                          E0 = 4153.5,
                          E1 = 12888.8,
                          A0 = 139500,
                          A1 = 2567000000000000000,
                          slope = 1.6,
                          Tf = 277)
{
  if(missing(HourTemp)) {
    stop("HourTemp must be present! Aborting.")
  }
  ## compute temperatures in Kelvin
  TK <- HourTemp + 273
  ## pre-compute some constants
  aa <- A0/A1
  ee <- E1-E0
  sr <- exp(slope*Tf*(TK-Tf)/TK)
  xi <- sr/(1+sr)
  xs <- aa*exp(ee/TK)
  eak1 <- exp(-A1*exp(-E1/TK))
  
  ## initialise PDBF with zero
  x=0
  for (l in c(2:length(HourTemp)))  {
    S <- x[l-1]
    if(x[l-1] >= 1) {
      S <- S*(1-xi[l-2])
    }
    x[l] <- xs[l-1]-(xs[l-1]-S)*eak1[l-1]
  }
  ## chill portions
  delta <- rep(0,length(HourTemp))
  ii <- which(x >= 1)
  delta[ii] <- x[ii]*xi[ii-1]
  if (summ) return(cumsum(delta))
  return(delta)
}

## x_s(t_i) according to the first reference
xsfct  <- function(A0, A1, E0, E1, temp) {
  return ( A0/A1 * exp(-(E0 - E1)/temp) )
}
## k_1(t_i) according to the first reference
k1fct <- function(A1, E1, temp) {
  return( A1*exp(-E1/temp) )
}
Weather_feature_generation = function(df,latitude = 43.00, cultivar = 'Riesling'){
  
  #Data filtering to exclude NAs and outliers
  df$date <- as.Date(df$date)
  df <- df %>% 
    arrange(date)
  df = filter(df, !is.na(Tmin), !is.na(Tmax))
  df$Year <- as.numeric(format(df$date, format = "%Y"))
  df$Month <-  as.numeric(format(df$date, format = "%m"))
  df$Day <-  as.numeric(format(df$date, format = "%d"))
  df <- filter(df, Month %in% c(1,2,3,4,8,9,10,11,12))
  # df <-df %>%
  #   group_by(Year) %>%
  #   mutate(ind = sum(is.na(Tmin))) %>%
  #   group_by(Year) %>%
  #   filter(!any(ind >= 10)) %>%
  #   select(-ind)
  #df <- filter(df, !is.na(Tmax) | is.na(Tmin), !is.na(Tmax) & !is.na(Tmin),!Tmax > 100, !Tmin < -100, !Tmin > Tmax)
  ##If the input df have fewer than 20 row left after filtering, the feature extraction could not be accomplished
  if (nrow(df) < 20) {
    df = NULL
  }
  
  else
  {
    
    #Estimate hourly temperature based on daily temperature variation
    data_all_hourly = stack_hourly_temps(df,latitude = latitude)[[1]]
    data_all_hourly$DOY = yday(data_all_hourly$date)
    data_all_hourly$Date = data_all_hourly$date
    data_all_hourly_1 = data.frame(Year = data_all_hourly$Year,
                                   Month = data_all_hourly$Month,
                                   Day = data_all_hourly$Day,
                                   DOY = data_all_hourly$DOY,
                                   Hour = data_all_hourly$Hour,
                                   Temp = data_all_hourly$Temp)
    
    
    #Chilling models computation
    CU <- chilling_units(data_all_hourly$Temp, summ = F)
    Utah<- modified_utah_model(data_all_hourly$Temp, summ = F)
    NC<-north_carolina_model(data_all_hourly$Temp, summ = F)
    GDH_10 <- GDH_linear(data_all_hourly_1, Tb = 10, Topt = 25, Tcrit = 36)
    GDH_7 <- GDH_linear(data_all_hourly_1, Tb = 7, Topt = 25, Tcrit = 36)
    GDH_4 <- GDH_linear(data_all_hourly_1, Tb = 4, Topt = 25, Tcrit = 36)
    GDH_0 <- GDH_linear(data_all_hourly_1, Tb = 0, Topt = 25, Tcrit = 36)
    DP <- Dynamic_Model(data_all_hourly$Temp, summ = F)
    
    
    CU = if_else(CU < 0, 0, CU)
    Utah = if_else(Utah < 0, 0 ,Utah)
    NC = if_else(NC < 0,0, NC)
    
    All_chilling_data <- data.frame(date = data_all_hourly$date,
                                    Month = format(data_all_hourly$date,format = "%b"),
                                    Year = format(data_all_hourly$date,format = "%Y"),
                                    CU = CU,
                                    Utah = Utah,
                                    NC = NC,
                                    DP = DP)
    
    All_chilling_data$Year  = as.numeric(All_chilling_data$Year)
    
    All_chilling_data$CU1 <- ifelse(All_chilling_data$Month %in% c('Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr'),All_chilling_data$CU, 0 )
    All_chilling_data$Utah1 <- ifelse(All_chilling_data$Month %in% c('Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr'),All_chilling_data$Utah, 0 )
    All_chilling_data$NC1 <- ifelse(All_chilling_data$Month %in% c('Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr'),All_chilling_data$NC, 0 )
    GDH_10$GDH <- ifelse(GDH_10$Month %in% c(1,2,3,4),GDH_10$GDH, 0 )
    GDH_7$GDH <- ifelse(GDH_7$Month %in% c(1,2,3,4),GDH_7$GDH, 0 )
    GDH_4$GDH <- ifelse(GDH_4$Month %in% c(1,2,3,4),GDH_4$GDH, 0 )
    GDH_0$GDH <- ifelse(GDH_0$Month %in% c(1,2,3,4),GDH_0$GDH, 0 )
    All_chilling_data$DP1 <- ifelse(All_chilling_data$Month %in% c('Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr'),All_chilling_data$DP, 0 )
    
    
    
    All_chilling_data$dormant_season <- ifelse(All_chilling_data$Month %in% c('Sep','Oct','Nov','Dec'), paste0(All_chilling_data$Year,'-',All_chilling_data$Year + 1),
                                               ifelse(All_chilling_data$Month %in% c('Jan','Feb','Mar','Apr'),paste0(All_chilling_data$Year -1,'-',All_chilling_data$Year),NA))
    
    All_chilling_data <- All_chilling_data[order(All_chilling_data$date),]
    
    
    
    Chilling_data_summary <-  All_chilling_data %>%
      group_by(dormant_season) %>%
      reframe(Utah= cumsum(Utah1),
                CU = cumsum(CU1),
                NC = cumsum(NC1),
                #GDD_10 = cumsum(GDD_10_1),
                #GDD_7 = cumsum(GDD_7_1),
                #GDD_4 = cumsum(GDD_4_1),
                #GDD_0 = cumsum(GDD_0_1),
                DP = cumsum(DP1),
                date = date)
    
    Chilling_data_summary_daily <-  Chilling_data_summary %>%
      group_by(date) %>%
      summarise(
        CU = max(CU),
        NC = max(NC),
        Utah = max(Utah),
        #GDD_10 = max(GDD_10),
        #GDD_7 = max(GDD_7),
        #GDD_4 = max(GDD_4),
        #GDD_0 = max(GDD_0),
        DP = max(DP)) %>%
      arrange(date)
    
    #GDHs computation
    GDHs = data.frame(date = as.Date(GDH_10$Date),
                      GDH10_1 = GDH_10$GDH,
                      GDH_7_1 = GDH_7$GDH,
                      GDH_4_1 = GDH_4$GDH,
                      GDH_0_1 = GDH_0$GDH)

    GDHs = GDHs %>%
      mutate(season = ifelse(format(GDHs$date,format = "%b") %in% c('Sep','Oct','Nov','Dec'), 
                             paste0(as.numeric(format(GDHs$date,format = "%Y")),"-",as.numeric(format(GDHs$date,format = "%Y")) + 1),
                             paste0(as.numeric(format(GDHs$date,format = "%Y")) - 1,"-",as.numeric(format(GDHs$date,format = "%Y"))))) %>%
      group_by(season) %>%
      reframe(GDH_10 =cumsum(GDH10_1),
                GDH_7 =cumsum(GDH_7_1),
                GDH_4 = cumsum(GDH_4_1),
                GDH_0 = cumsum(GDH_0_1),
                date = date) %>%
      arrange(date) %>%
      select(-season)
    
    
    #Combined chilling models and GDHs
    cumulative_feature_summary_daily <- merge(Chilling_data_summary_daily,GDHs, by = c('date'))
    
    
    #Daily temperature descriptors
    data_all_daily_T <-  data_all_hourly  %>%
      group_by(date) %>%
      summarise(#ID = station[1],
        lat = latitude,
        #lon = lon[1],
        #ST = State[1],
        Year = mean(Year),
        Month = mean(Month),
        Day = mean(Day),
        min_temp = min(Temp),
        max_temp = max(Temp),
        within_day_range_temp = max(Temp) - min(Temp),
        Naive_average_temp = mean(Temp),
        Median_temp = median(Temp))
    
    
    #REWMAs
    window_size = c(2,4,6,8,10,12,14,16,18,20)
    
    #min
    min_data_number = length(inverse_EWA(data_all_daily_T$min_temp, window = window_size[2]))
    min_REWA = data.frame(NA_col = rep(NA, min_data_number))
    for (i in 1:length(window_size)) {
      vname = print(paste0('min_REWMA_',window_size[i],"day"))
      #assign(vname, inverse_EWA(data_all_daily_T$min_temp, window = window_size[i]))
      #col = inverse_EWA(data_all_daily_T$min_temp, window = window_size[i])
      min_REWA[,i] = inverse_EWA(data_all_daily_T$min_temp, window = window_size[i])
      colnames(min_REWA)[i] = vname
    }
    
    #max
    max_data_number = length(inverse_EWA(data_all_daily_T$max_temp, window = window_size[2]))
    max_REWA = data.frame(NA_col = rep(NA, max_data_number))
    for (i in 1:length(window_size)) {
      vname = print(paste0('max_REWMA_',window_size[i],"day"))
      #assign(vname, inverse_EWA(data_all_daily_T$max_temp, window = window_size[i]))
      #col = inverse_EWA(data_all_daily_T$max_temp, window = window_size[i])
      max_REWA[,i] = inverse_EWA(data_all_daily_T$max_temp, window = window_size[i])
      colnames(max_REWA)[i] = vname
    }
    
    #mean
    mean_data_number = length(inverse_EWA(data_all_daily_T$Naive_average_temp, window = window_size[2]))
    mean_REWA = data.frame(NA_col = rep(NA, mean_data_number))
    for (i in 1:length(window_size)) {
      vname = print(paste0('mean_REWMA_',window_size[i],"day"))
      #assign(vname, inverse_EWA(data_all_daily_T$mean_temp, window = window_size[i]))
      #col = inverse_EWA(data_all_daily_T$mean_temp, window = window_size[i])
      mean_REWA[,i] = inverse_EWA(data_all_daily_T$Naive_average_temp, window = window_size[i])
      colnames(mean_REWA)[i] = vname
    }
    
    
    
    
    #EWMAs
    
    #min
    min_data_number = length(movavg(data_all_daily_T$min_temp, n = window_size[2],type = 'e'))
    min_EWA = data.frame(NA_col = rep(NA, min_data_number))
    for (i in 1:length(window_size)) {
      vname = print(paste0('min_EWMA_',window_size[i],"day"))
      min_EWA[,i] = movavg(data_all_daily_T$min_temp, n = window_size[i],type = 'e')
      colnames(min_EWA)[i] = vname
    }
    
    
    #max
    max_data_number = length(movavg(data_all_daily_T$max_temp, n = window_size[2],type = 'e'))
    max_EWA = data.frame(NA_col = rep(NA, max_data_number))
    for (i in 1:length(window_size)) {
      vname = print(paste0('max_EWMA_',window_size[i],"day"))
      max_EWA[,i] = movavg(data_all_daily_T$max_temp, n = window_size[i],type = 'e')
      colnames(max_EWA)[i] = vname
    }
    
    
    #mean
    mean_data_number = length(movavg(data_all_daily_T$Naive_average_temp, n = window_size[2],type = 'e'))
    mean_EWA = data.frame(NA_col = rep(NA, mean_data_number))
    for (i in 1:length(window_size)) {
      vname = print(paste0('mean_EWMA_',window_size[i],"day"))
      mean_EWA[,i] = movavg(data_all_daily_T$Naive_average_temp, n = window_size[i],type = 'e')
      colnames(mean_EWA)[i] = vname
    }
    
    
    
    
    #Output
    df_out = merge(data_all_daily_T, cumulative_feature_summary_daily, by = c('date'))
    df_out = data.frame(df_out,
                        min_EWA,
                        min_REWA,
                        max_EWA,
                        max_REWA,
                        mean_EWA,
                        mean_REWA)
    
    df_out = filter(df_out, !Month == 8)
    
    
    #Add cultivar features in the df
    selected_cultivar = cultivar 
    selected_cultivar_colname = paste0('Cultivar.',selected_cultivar)

    df_out$season <-  ifelse(df_out$Month %in% c(9,10,11,12), paste0(df_out$Year,'-',df_out$Year + 1),
                         ifelse(df_out$Month %in% c(1,2,3,4),paste0(df_out$Year -1,'-',df_out$Year),NA))
    df_out <- filter(df_out,!is.na(season))
    df_out[Cultivars] = 0
    col_number = which(colnames(df_out) == selected_cultivar_colname)
    df_out[,col_number] = 1
    
    
    #Add Days_in_season in the df
    df_out$Days_in_season = as.numeric(df_out$date - if_else(as.numeric(month(df_out$date)) %in% c(9,10,11,12), ymd(paste0(year(df_out$date),"-",9,"-",01)),
                                                                                 ymd(paste0(year(df_out$date)-1,"-",9,"-",01))))
    
    #Deleted unnecessary columns
    colnames(df_out)[which(colnames(df_out) == 'date')] = 'Date'
    df_out = df_out[,-which(colnames(df_out) %in% c('lat','Year','Month','Day','DP','season','Median_temp'))]
    
    #Output 
    return(df_out)
  }
  
}




#This is an example of feature extraction. You need to identify the latitude of the vineyard and the cultivar for the feature extraction.
cultivars_to_choose_from
all_data_feature_extracted <- Weather_feature_generation(all_data,latitude = latitude, cultivar = cultivar)

write_csv(all_data_feature_extracted, file = input_file)


