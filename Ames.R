# Ames Project
# 1. EDA: Understand and Clean Data
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building (1. Linear Regression, ,2. Decision Trees, 3. Random Forests)
# 5. Model Deployment & RMSE comparison between 3 models


# Code begins -------------------------------------------------------------

# Loading the library -----------------------------------------------------
library(tidyverse)
library(Amelia)
library(mice)
library(janitor)
library(corrplot)
library(randomForest)
library(caret)
library(rpart)

setwd("/Users/jaymehdi/Iowa_advanced_regression")

# Reading the data file ---------------------------------------------------

df<- read_csv("merge_data_update.csv")

#removing X1
df$X1 <- NULL

# Basic exploration
summary(df)
str(df)



# Feature Engineering -----------------------------------------------------


# Checking for missing values
naniar::gg_miss_var(df)

# There are total 7 columns with >40% missing values in the dataset
df %>% summarize_all(funs(sum(is.na(.)) / length(.)))%>% 
  transpose() %>% 
  unlist() %>% 
  data.frame(missing_pct=.) %>% 
  arrange(desc(missing_pct))

#Removing column with >40%+ missing values
df_2 <- df %>% 
  purrr::discard(~sum(is.na(.x))/length(.x)* 100 >=33)

#Removing rows with greater than 10% missing values
df_3 <- df_2[which(rowMeans(!is.na(df_2)) > 0.33), ] %>% 
  clean_names()# No rows removed


# Imputing missing values in the remaining ones
df_2 %>% summarize_all(funs(sum(is.na(.)) / length(.)))%>% 
  transpose() %>%  
  unlist() %>% 
  data.frame(missing_pct=.) %>% 
  arrange(desc(missing_pct)) %>% 
  filter(missing_pct>0) 

# Imputing missing values

com <- mice(df_3, m=1, method='cart',printFlag=FALSE)
df_4 <- na.omit(complete(com))

#Checking to see number of missing values now
sum(sapply(df_4, function(x) { sum(is.na(x)) }))


# Visualization  ----------------------------------------------------------
`%!in%` = Negate(`%in%`)


df_4 %>% 
 select_if(is.numeric) %>% 
  gather(key="key",
         value="value") %>% 
  filter(key %in% c("lot_frontage","lot_area","overall_qual","garage_area",
                    "fireplaces","poo_area","mas_vnr_area","bedroom_abv_gr",
                    "full_bath","gla")) %>% 
  ggplot(aes(x="", value)) + 
  geom_boxplot() +
  facet_wrap(~key,scales="free")+
  theme(axis.text.x = element_text(angle = 90, hjust =1)) +
  xlab('Important Numeric columns')+
  theme_light()

df_4 %>% 
  select_if(is.numeric) %>% 
  gather(key="key",
         value="value") %>% 
  filter(key %in% c("lot_frontage","lot_area","overall_qual","garage_area",
                    "fireplaces","poo_area","mas_vnr_area","bedroom_abv_gr",
                    "full_bath","gla")) %>% 
  ggplot(aes(x=value)) + 
  geom_histogram() +
  facet_wrap(~key,scales="free")+
  theme(axis.text.x = element_text(angle = 90, hjust =1)) +
  xlab('Important Numeric columns')+
  theme_light()

# Converting numeric var to log basis above, adding to avoid Inf in the output
df_4$lot_area <- log(df_4$lot_area+1) 
df_4$lot_frontage <- log(df_4$lot_frontage+1) 
df_4$mas_vnr_area <- log(df_4$mas_vnr_area+1) 


# Correlation with sale_price


all_numVar <-  df_4 %>% select_if(is.numeric)
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'sale_price'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]


corrplot(cor_numVar, tl.col="black", 
         tl.pos = "lt", tl.cex = 0.7,cl.cex = .7,
         number.cex=.7,method = "number",
        type="upper",title = "Correlation of variable with Sale price (Highest to lowest)")


# Finding Important variable -------------------------------------------------------

set.seed(2018)
quick_RF <- randomForest(x=df_4[,-3], y=df_4$sale_price, ntree=100,importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) +
  geom_bar(stat = 'identity') + 
  labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + 
  coord_flip() + theme(legend.position="none")

# Recoding Important categorical variable

# bsmt_qual
# ms_zoning
# house_style
# functional
# bsmt_fin_type1

df_5 <- df_4 %>% 
  mutate(bsmt_qual_new=case_when(bsmt_qual=="Ex"~1,
                             bsmt_qual=="Fa"~2,
                             bsmt_qual=="Gd"~3,
                             bsmt_qual=="Po"~4,
                             bsmt_qual=="TA"~5,
                             TRUE ~0),
         ms_zoning_new=case_when(ms_zoning=="C (all)"~1,
                                 ms_zoning=="FV I (all)"~2,
                                 ms_zoning=="RH"~3,
                                 ms_zoning=="RL"~4,
                                 ms_zoning=="RM"~5,
                                 TRUE ~ 0),
         house_style_new=case_when(house_style=="1.5Fin"~1,
                                   house_style=="1.5UnF"~2,
                                   house_style=="1Story"~3,
                                   house_style=="2.5Fin"~4,
                                   house_style=="2.5UnF"~5,
                                   house_style=="2Story"~6,
                                   house_style=="SFoyer"~7,
                                   house_style=="SLvl"~8,
                                   TRUE ~0),
         functional_new=case_when(functional=="Maj1"~1,
                                  functional=="Maj2"~2,
                                  functional=="Min1"~3,
                                  functional=="Min2"~4,
                                  functional=="Mod"~5,
                                  functional=="Sal"~6,
                                  functional=="Typ"~7,
                                  TRUE ~0),
         bsmt_fin_type1_new=case_when(bsmt_fin_type1=="ALQ"~1,
                                      bsmt_fin_type1=="BLQ"~2,
                                      bsmt_fin_type1=="GLQ"~3,
                                      bsmt_fin_type1=="LwQ"~4,
                                      bsmt_fin_type1=="Rec"~5,
                                      bsmt_fin_type1=="Unf"~6,
                                      TRUE ~ 0)) %>% 
  select(  -  bsmt_qual,
           - ms_zoning,
           -  house_style,
           - functional,
           - bsmt_fin_type1)


# Model Building ----------------------------------------------------------

`%!in%` = Negate(`%in%`)


#Creating train & test sets

set.seed(2021)
index <- createDataPartition(df_5$sale_price,p=0.7,list=FALSE)
train <- df_5[index,]
test <- df_5[-index,]

# Creating the formula to be used in models
#Formula(remove Id)
col.names <- colnames(train)
col.names <- col.names[col.names %!in%  c('pid',"sale_price","condition2",
                                          "exter_cond","heating_qc","zng_cd_pr",
                                          "roof_matl","garage_qual","prop_addr",
                                          "class_pr_s","class_sc_s")]
fmla <- as.formula(paste("sale_price ~ ", paste(col.names, collapse= "+")))


# Linear Regression -------------------------------------------------------

#training linear regression Model
lm_model = lm(formula = fmla, data = train)
lm_model.predict <- predict(lm_model,test)

# RMSE from the Linear regression model
rmse_lreg <- RMSE(test$sale_price, lm_model.predict)


# Decision Trees ----------------------------------------------------------

#training linear regression Model
tree_model = rpart(formula = fmla, data = train)
tree_model.predict <- predict(tree_model,test)

# RMSE from the Linear regression model
rmse_tree <- RMSE(test$sale_price, tree_model.predict)


# Random Forest -----------------------------------------------------------

#training Random Forest Model
rf_model = randomForest(formula = fmla, data = train)
rf_model.predict <- predict(rf_model,test)

# RMSE from the Linear regression model
rmse_rf <- RMSE(test$sale_price, rf_model.predict)



# RMSE Comparison ---------------------------------------------------------

paste('RMSE for LM model is ',round(rmse_lreg,1))
paste('RMSE for Decision tree model is ',round(rmse_tree,1))
paste('RMSE for Randomf Forest model is ',round(rmse_rf,1))

# Linear Regression gives us the best RMSE followed by 
# RF model 
# and RMSE on decision tree is 3rd among 3 models


