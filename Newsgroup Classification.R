###### CS6405 ######
###### Data Mining ######
###### Shubham Kajaria 119220254 ######
###### Please set the path before running the program ######

rm(list = ls())
library(stringr)
library(readr)
library(class)
library(randomForest)
library(tree)
library(e1071)
library(caret)
library(tm)
library(textstem)
library(mlr)
library(ggplot2)
library(reshape2)

path = "/Users/shubham/Desktop/DataMining/Newsgroups/"




###### Storing all the directories in the avaiable given path by leaving the base directory ######
total_dir = list.dirs(path)
total_dir = total_dir[2: length(total_dir)]

###### Storing all the files from each of the directories ######
all_files = all_class = c()
i = 1
for( dir in total_dir ){
  for( f in cbind(list.files(dir) )){
    all_files[i] = file.path(dir, f)
    all_class[i] = basename(dir)
    i = i+1
  }
}




###################################### Exploring Data #####################################

###### Sort the dataframe as per the descending word frequency ######
sort_df = function(all_files){
  full_data = c()
  for(i in 1:length(all_files)){
    data = read_file(all_files[i])
    full_data = paste0(full_data,data,sep=" ")
  }
  full_data = gsub("[[:punct:][:blank:]]+|[\t\r\n]", " ", full_data)
  total_words = strsplit(full_data, " ")[[1]]
  tb = table(total_words)
  parsed_df = data.frame(UniqueWords = names(tb),Count = as.vector(tb))
  parsed_df$UniqueWords = as.character(parsed_df$UniqueWords)
  sorted_df = parsed_df[order(parsed_df$Count, decreasing = TRUE), ]
  return(sorted_df)
}

###### TOP 200 words ######
sorted_df = sort_df(all_files)
row.names(sorted_df) = 1:nrow(sorted_df)
# Removed spaces which is present at the first position
cbind(Words = sorted_df$UniqueWords[2:201], Count = sorted_df$Count[2:201])

###### TOP 200 words length wise between 4 and 20 ######
cal_top_words_lengthwise = function(sorted_df, start_l = 4, end_l = 20, total_w = 200){
  count_word = 1; word_new = c(); count_freq = c();
  for(i in 1:length(sorted_df$UniqueWords)){
    w = sorted_df$UniqueWords[i]
    if(str_length(w) >= start_l & str_length(w) <= end_l & count_word <= total_w){
      word_new = c(word_new, w)
      count_freq = c(count_freq, sorted_df$Count[i])
      count_word = count_word + 1
    }
  }
  return(list(word_new, count_freq))
}
ttl_words_freq = cal_top_words_lengthwise(sorted_df, 4, 20, 200)
cbind(Words = ttl_words_freq[[1]], Count = ttl_words_freq[[2]])




############################### Basic Evaluation Starts Here ##############################

###### Function returns a list of words and their frequencies ######
parse_lines = function(data){
  ldata = lcount = lclass = c()
  item = 1
  if(data == ""){
    next
  }
  total_words = strsplit(data, " ")[[1]]
  for(j in 1:length(total_words)){
    word = total_words[j]
    if(word == ""){
      next
    }
    if(!word %in% ldata){
      ldata[item] = word
      lcount[item] = 1
      item = item + 1
    }
    else{
      index = which(ldata == word)
      lcount[index] = lcount[index] + 1
    }
  }
  return(list(ldata, lcount))
}

###### This function (Pre-processing) removes the unnecessary data ######
remove_unnecessary_data = function(data){
  data = tolower(data)
  data = trimws(gsub("\\w*[0-9]+\\w*\\s*", "", data))
  data = removeWords(data, stopwords("en"))
  data = lemmatize_strings(data)
  data = gsub("[[:punct:][:blank:]]+|[\t\r\n]", " ", data)
  data = gsub('\\b\\w{1,3}\\b','',data)
  return(data)
}

###### This function returns file wise unique words & their counts ######
run_allfiles = function(all_files, tune = 1){
  list_data = c(); lclass = c();
  df = data.frame(matrix(ncol = 4, nrow = 0), stringsAsFactors = FALSE)
  df1 = df
  
  for(i in 1:length(all_files)){
    data = read_file(all_files[i])
    data = gsub("[[:blank:]+|[\t\r\n\"]", " ", data)
    if(tune == 1){
      data = remove_unnecessary_data(data)
    }
    
    list_data = parse_lines(data)
    df1 = data.frame(list_data[[1]], list_data[[2]], all_files[i],
                     basename(dirname(all_files[1])), stringsAsFactors = FALSE)
    df = rbind(df, df1)
  }
  names(df) = c("UniqueWords", "Count", "Files", "CLASS")
  df$Count = as.numeric(df$Count)
  df$CLASS = as.factor(df$CLASS)
  return(df)
}

###### Calculate class wise probability of the words present in the test set ######
class_wise_prob = function(df_trainset, df_prob, test_sent_words, l){
  lv = levels(as.factor(df_trainset$CLASS))
  s = sum(df_trainset[df_trainset$CLASS == lv[l], -which(names(df_trainset) == "CLASS")])
  ttl = length(names(df_trainset)) - 1
  
  for(word in test_sent_words){
    check = df_trainset[1, word]
    if(length(check) > 0){
      wc = sum(df_trainset[df_trainset$CLASS == lv[l], word])
      df_prob[l, word] = (wc + 1) / (s + ttl)
    }else{
      df_prob[l, word] = 1 / (s + ttl)
    }
  }
  return(df_prob)
}

###### Calculate probability of the files in the testset ######
probability_fn = function(df_trainset, df_testset){
  counts_df = df_testset[, -which(names(df_testset) == "CLASS")]
  index = which(counts_df >= 1, arr.ind = TRUE)
  dummy_df = data.frame(index)
  all_words = names(df_testset)[unique(dummy_df$col)]
  
  index1 = which( df_trainset >= 1, arr.ind = TRUE)
  dummy_df1 = data.frame(index1)
  df_trainset = df_trainset[, names(df_trainset)[unique(dummy_df1$col)]]
  df_prob = data.frame(matrix(0, ncol = length(all_words), 
                              nrow = 4),stringsAsFactors = FALSE)
  names(df_prob) = all_words
  df_prob = class_wise_prob(df_trainset, df_prob, all_words, 1)
  df_prob = class_wise_prob(df_trainset, df_prob, all_words, 2)
  df_prob = class_wise_prob(df_trainset, df_prob, all_words, 3)
  df_prob = class_wise_prob(df_trainset, df_prob, all_words, 4)
  return(df_prob)
}

###### Function returns the class having max probability for a test file ######
check_file_class = function(test_sent_words, df_prob){
  logprob1 = logprob2 = logprob3 = logprob4 = 0
  logprob1 = sum(log(df_prob[1, test_sent_words]))
  logprob2 = sum(log(df_prob[2, test_sent_words]))
  logprob3 = sum(log(df_prob[3, test_sent_words]))
  logprob4 = sum(log(df_prob[4, test_sent_words]))
  prob_file = c(logprob1, logprob2, logprob3, logprob4)
  return(which.max(prob_file))
}

###### Function create a single dataframe having each column as a unique word ######
words_as_columns = function(df, files){
  df_new = data.frame(matrix(0, ncol = length(unique(df$UniqueWords)), 
                             nrow = length(files)),stringsAsFactors = FALSE)
  names(df_new) = unique(df$UniqueWords)
  df_new$CLASS = rep('', length(files))
  
  for(i in 1:length(files)){
    index = which(df$Files == files[i])
    word = df$UniqueWords[index]
    df_new[i, word] = df$Count[index]
    df_new$CLASS[i] = basename(dirname(files[i]))
  }
  return(df_new)
}

###### This function displays performance matrices ######
performance_matrices = function(prob_file, test_class){
  # Confusion Matrix
  lv = levels(as.factor(test_class))
  conf_m = suppressWarnings(confusionMatrix(data = as.factor(lv[prob_file]), reference = as.factor(test_class)))
  print(conf_m)
  
  out = melt(as.matrix(conf_m))
  names(out) = c("Predicted_Response", "Actual_Response", "value")
  ggplot(data=out, aes(x=Predicted_Response, y=Actual_Response, fill=value)) + 
    geom_tile() + 
    geom_text(aes(label=value), color='black', size = 5) +
    theme(text = element_text(size = 10)) +
    theme(axis.text = element_text(size = 10)) +
    theme(axis.title = element_text(size = 13)) +
    labs(title='Confusion Matrix') +
    scale_fill_distiller(palette="Blues", direction=1) + 
    coord_equal()
  
  # Precision
  print(cbind(Precision = conf_m$byClass[, "Precision"]))
  
  # Recall
  print(cbind(Recall = conf_m$byClass[, "Recall"]))
  
  # F1-score
  print(cbind(F1score = conf_m$byClass[, "F1"]))
}

###### Creating a 70:30 train-test split ######
create_test_train_split = function(all_files, all_class){
  set.seed(983)
  index = sample(1:length(all_files), round(0.7 * length(all_files)), replace = FALSE)
  
  # Train set
  train_files = all_files[index]
  train_class = all_class[index]
  
  # Test set
  test_files = all_files[-index]
  test_class = all_class[-index]
  return(list(index, test_class))
}

###### Making the first letter of all Columns to Upper case(Better Readability) ######
CapStr <- function(y) {
  c <- strsplit(y, " ")[[1]]
  paste(toupper(substring(c, 1,1)), substring(c, 2),
        sep="", collapse=" ")
}

###### Create a bag of words ######
create_words_df = function(all_files, tune = 1){
  print("Creating bag of words......")
  df_full_row = run_allfiles(all_files, tune)
  df_full = words_as_columns(df_full_row, all_files)
  name = names(df_full)
  names(df_full) = sapply(name, CapStr)
  return(df_full)
}

###### Call this function to create a test train split ######
l = create_test_train_split(all_files, all_class)
index_sample = l[[1]]
test_class = l[[2]]




##################################### UNPROCESSED DATA ####################################

###### Call this function with (tune=0) for creating a dataframe of UNPROCESSED DATA ######
tune = 0
returned_df = create_words_df(all_files, tune)

###### Create a trainset and testset dataframe ######
df_trainset = returned_df[index_sample, ]
df_testset = returned_df[-index_sample, ]

###### Initialise the below variables to create a dataframe of UNPROCESSED DATA ######
###### It will be used for Basic evaluation using different models ######
x.train = df_trainset[, -which(names(df_trainset) == "CLASS")]
y.train = df_trainset$CLASS
x.test = df_testset[, -which(names(df_testset) == "CLASS")]
y.test = df_testset$CLASS




################################ Applying Naive Bayes #####################################
call_naive_bayes = function(df_trainset, df_testset){
  print("Naive Bayes is running......")
  prob_file = c()
  df_prob = probability_fn(df_trainset, df_testset)
  for(i in 1:nrow(df_testset)){
    counts = df_testset[i, -which(names(df_testset) == "CLASS")]
    test_sent_words = names(df_testset)[which(counts > 0)]
    prob_file[i] = check_file_class(test_sent_words, df_prob)
  }
  return(prob_file)
}

###### Call the Naive Bayes model to train and test ######
prob_file = call_naive_bayes(df_trainset, df_testset)

###### Call this function to calculate the performance matrices ######
performance_matrices(prob_file, df_testset$CLASS)




################################ Applying KNN ############################################
apply_knn = function(x.train, y.train, x.test){
  print("KNN is running......")
  knno = knn(train = x.train, test = x.test, cl = y.train)
  # Call this function to calculate the performance matrices
  plot(knno, col = "dodgerblue3")
  return(knno)
}

set.seed(1)
knno = apply_knn(x.train, y.train, x.test)

###### Call this function to calculate the performance matrices ######
performance_matrices(knno, y.test)




################################ Applying Random Forest ###################################
apply_rf = function(x.train, y.train, x.test){
  print("Random Forest is running......")
  rf = randomForest(x.train, as.factor(y.train), ntree = 200)
  pred_rf = predict(rf, newdata = x.test)
  return(list(pred_rf, rf))
}

set.seed(1)
rf_out = apply_rf(x.train, y.train, x.test)

###### Call this function to calculate the performance matrices ######
pred_rf = rf_out[[1]]
performance_matrices(pred_rf, y.test)

###### Plotting the misclassification error rate
rf = rf_out[[2]]
oob.err.data = data.frame(Trees = rep(1:nrow(rf$err.rate), times = 5),
                          Type = rep(c("OOB", "comp.sys.ibm.pc.hardware", "sci.electronics", 
                                       "talk.politics.guns", "talk.politics.misc"),each = nrow(rf$err.rate)),
                          Error = c(rf$err.rate[, "OOB"], 
                                    rf$err.rate[, "comp.sys.ibm.pc.hardware"],
                                    rf$err.rate[, "sci.electronics"],
                                    rf$err.rate[, "talk.politics.guns"],
                                    rf$err.rate[, "talk.politics.misc"]))
ggplot(data=oob.err.data, aes(x=Trees, y=Error))+
  geom_line(aes(color=Type))
################################ Basic evaluation Ends Here ###############################




############################## Robust Evaluation Starts Here ##############################




###### Call this function to return the 1500 important words ######
###### Length of the words are between 4 & 10 ######
feature_selection = function(sorted_df, df_trainset){
  imp_words = cal_top_words_lengthwise(sorted_df, 4, 10,1500)[[1]]
  imp_words = remove_unnecessary_data(imp_words)
  imp_words = imp_words[imp_words!='']
  imp_words = as.vector(sapply(imp_words, CapStr))
  index = which(!imp_words %in% names(df_trainset))
  imp_words = imp_words[-index]
  imp_words[length(imp_words) + 1] = "CLASS"
  return(imp_words)
}




##################################### PROCESSED DATA ####################################

###### Call this function with tune = 1 for creating a dataframe of processed data ######
tune = 1
returned_df = create_words_df(all_files, tune)




###### Create a trainset and testset dataframe ######
df_trainset = returned_df[index_sample, ]
df_testset = returned_df[-index_sample, ]




###### Applying Feature Selection, extracting important words ######
imp_words = feature_selection(sorted_df, df_trainset)




###### Tuning KNN with mlr ######
traintask = makeClassifTask(data = df_trainset[, imp_words], target = "CLASS")
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc('CV', iters = 10L)

ps.knn = makeParamSet(makeDiscreteParam("k", values = seq(1,30,1)))
res.knn = tuneParams("classif.kknn", task = traintask, resampling = rdesc, 
                 par.set = ps.knn, control = ctrl, measures = acc)
###### [Tune] Result: k=7 : acc.test.mean = 0.5535714 ######

###### Plot KNN optimal 'k' value ######
data = generateHyperParsEffectData(res.knn)
plotHyperParsEffect(data, x = "iteration", y = "acc.test.mean",
                    plot.type = "line")




###### Tuning Random Forest with mlr ######
ps.rf = makeParamSet(makeDiscreteParam("ntree", values = seq(50, 200, 50)),
                     makeDiscreteParam("mtry", values = seq(50, 250, 50)))
res.rf = tuneParams("classif.randomForest", task = traintask, resampling = rdesc, 
                 par.set = ps.rf, control = ctrl, measures = acc)
###### [Tune] Result: ntree=100; mtry=200 : acc.test.mean = 0.9678571 ######




###### Tuning SVM with mlr ######
kernels_model = c("vanilladot", "polydot")
ps.svm = makeParamSet(
  makeLogicalParam("scaled"),
  makeDiscreteParam("kernel", values = kernels_model),
  makeDiscreteParam("C", values = c(0.01, 0.05, 0.1,0.5,1))
  )
res.svm = tuneParams("classif.ksvm", task = traintask, resampling = rdesc, 
                     par.set = ps.svm, control = ctrl, measures = acc)
###### [Tune] Result: scaled=FALSE; kernel=vanilladot; C=0.01 : acc.test.mean=0.8392857 ######




###### Tuning Decision Tree with mlr ######
ps.tree = makeParamSet(makeDiscreteParam("minsplit", values = c(1,5,10,15,20)),
                       makeDiscreteParam("maxdepth", values = c(10,20,30)),
                       makeDiscreteParam("cp", values = c(0.0001,0.001, 0.01)))
res.tree = tuneParams("classif.rpart", task = traintask, resampling = rdesc, par.set = ps.tree, 
                      control =ctrl, measures = acc)
###### [Tune] Result: minsplit=20; maxdepth=30; cp=0.001 : acc.test.mean=0.9714286 ######




################################## Generate Learning Data #################################

###### Observe how the performance changes with an increasing number of observations ######

lrns = list(
  makeLearner("classif.knn", id = "KNN", k = res.knn$x$k),
  makeLearner("classif.randomForest", id = "RF", 
              ntree = res.rf$x$ntree, mtry= res.rf$x$mtry),
  makeLearner("classif.ksvm", id = "SVM", scaled = res.svm$x$scaled, 
              kernel=res.svm$x$kernel, C=res.svm$x$C),
  makeLearner("classif.rpart", id = "Tree", 
              minsplit = res.tree$x$minsplit, 
              maxdepth= res.tree$x$maxdepth, cp=res.tree$x$cp)
)

rin = makeResampleDesc(method = "CV", iters = 5)
lc = generateLearningCurveData(learners = lrns, task = traintask, 
                               percs = seq(0.1, 1, by = 0.1), 
                               measures = acc,
                               resampling = rin)
plotLearningCurve(lc)




############## Calculate Accuracy on Testing set after Robust implementation ##############
testtask = makeClassifTask(data = df_testset[, imp_words], target = "CLASS")

###### Test the Naive Bayes model for the formatted data ######
prob_file = call_naive_bayes(df_trainset[, imp_words], df_testset[, imp_words])
performance_matrices(prob_file, df_testset$CLASS)




###### Calculate Accuracy of Testing set on  KNN ######
set.seed(1)
knn1 = makeLearner("classif.kknn", k = res.knn$x$k)
model = mlr::train(knn1, traintask)
predict_knn <- predict(model, testtask)
performance_matrices(predict_knn$data[, 3], df_testset$CLASS)




###### Calculate Accuracy of Testing set on Random Forest ######
set.seed(1)
rf1 = makeLearner("classif.randomForest", ntree = res.rf$x$ntree, mtry = res.rf$x$mtry)
model = mlr::train(rf1, traintask)
predict_rf <- predict(model, testtask)
performance_matrices(predict_rf$data[, 3], df_testset$CLASS)




###### Calculate Accuracy of Testing set on SVM ######
set.seed(1)
svm = makeLearner("classif.ksvm", C = res.svm$x$C, kernel = res.svm$x$kernel, scaled = res.svm$x$scaled)
model = mlr::train(svm, traintask)
predict_svm <- predict(model, testtask)
performance_matrices(predict_svm$data[, 3], df_testset$CLASS)




###### Calculate Accuracy of Testing set on Decision Tree ######
set.seed(1)
tree.out = makeLearner("classif.rpart", minsplit = res.tree$x$minsplit, maxdepth = res.tree$x$maxdepth,
                       cp = res.tree$x$cp)
model = mlr::train(tree.out, traintask)
predict_tree <- predict(model, testtask)
performance_matrices(predict_tree$data[, 3], df_testset$CLASS)

plotLearnerPrediction(knn1, traintask)
