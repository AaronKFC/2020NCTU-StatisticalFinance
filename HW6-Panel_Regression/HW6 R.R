require(dplyr)
require(tidyverse)
require(magrittr)
require(readr)
require(plm)
require(ggplot2)
require(stargazer)

data <- read.csv("C:\\Users\\Yoga\\Desktop\\dairy.csv"
                 , header = T)[, c('FARM', 'YEAR', 'MILK', 'COWS', 'LAND' ,'LABOR', 'FEED')]
head(data)
summary(data)
plot(data[, 3:length(data)], main = 'Scatter Plot')
data %>% summarise_all(funs(class)) # 看資料屬性

pdata=pdata.frame(data,index = c('FARM','YEAR'))
pdata %>% 
  group_by(FARM) %>% #依FARM分組進行以下程序：
  mutate(
    COWS_demean=COWS-mean(COWS),
    LAND_demean=LAND-mean(LAND),
    LABOR_demean=LABOR-mean(LABOR),
    FEED_demean=FEED-mean(FEED),
    logMILK=log(MILK)
  ) %>%
  select(COWS_demean,LAND_demean,LABOR_demean,FEED_demean,logMILK) %>%
  ungroup() -> demean_results # grouping variable會被保留

demean_results %>%
  ggplot()+
  geom_point(aes(x=COWS_demean,y=logMILK))+
  geom_smooth(aes(x=COWS_demean,y=logMILK),method = "lm",se=FALSE)

pool=log(MILK) ~ COWS + LAND + LABOR + FEED + FARM + YEAR
poolmodel<-plm(pool, data=pdata, model='pooling')
summary(poolmodel)

indmodel<-plm(pool, model='within', effect='individual',data = pdata)
summary(indmodel)

timemodel<-plm(pool, model='within', effect='time',data = pdata)
summary(timemodel)

bothmodel<-plm(pool, model='within', effect='twoways',data = pdata)
summary(bothmodel)

stargazer(poolmodel,indmodel,timemodel,bothmodel,type='text',
          column.labels = c("Pooled OLS","FE-individual","FE-time","FE-two-ways"))