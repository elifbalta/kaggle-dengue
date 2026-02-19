
# Dengue Infection Classification Project
# Author: Elif Balta
# Description: Machine learning pipeline built for STAT412 Kaggle competition

#Step 1 : Load required libraries
library(tidyverse)       
library(forcats)
library(caret)       
library(randomForest)
library(xgboost)
library(pROC)   

set.seed(412)  #for reproducibility


#Step 2: Read the training and test datasets

train_df <- read_csv("data/train_df.csv")
test_df  <- read_csv("data/test_df.csv")

#Step 3: Feature engineering

#Convert Outcome to factor with levels "Negative"/"Positive"
train_df <- train_df %>%
  mutate(
    Outcome = factor(Outcome, levels = c(0, 1), labels = c("Negative", "Positive"))
  )

#Create AgeGroup2 in both train and test
train_df <- train_df %>%
  mutate(
    AgeGroup2 = cut(
      Age,
      breaks = c(-Inf, 12, 18, 35, 60, Inf),
      labels = c("Child", "Teen", "YoungAdult", "Adult", "Senior")
    )
  )

test_df <- test_df %>%
  mutate(
    AgeGroup2 = cut(
      Age,
      breaks = c(-Inf, 12, 18, 35, 60, Inf),
      labels = c("Child", "Teen", "YoungAdult", "Adult", "Senior")
    )
  )

#Create Area_House interaction feature in both train and test
train_df <- train_df %>%
  mutate(
    Area_House = paste(AreaType, HouseType, sep = "_")
  )

test_df <- test_df %>%
  mutate(
    Area_House = paste(AreaType, HouseType, sep = "_")
  )

#List of categorical columns to convert and align
cats <- c("Gender", "Area", "AreaType", "HouseType", "District", "AgeGroup2", "Area_House")

#In both train and test, replace NA in each categorical with "Unknown" and convert to factor
for (p in cats) {
  if (p %in% names(train_df)) {
    train_df[[p]] <- as.character(train_df[[p]])
    train_df[[p]][is.na(train_df[[p]])] <- "Unknown"
    train_df[[p]] <- factor(train_df[[p]])
  }
  if (p %in% names(test_df)) {
    test_df[[p]] <- as.character(test_df[[p]])
    test_df[[p]][is.na(test_df[[p]])] <- "Unknown"
    test_df[[p]] <- factor(test_df[[p]])
  }
}

#Align factor levels of test to train for each categorical column
for (p in cats) {
  if (p %in% names(train_df) && p %in% names(test_df)) {
    train_levels <- levels(train_df[[p]])
    test_vals    <- as.character(test_df[[p]])
    test_vals[!(test_vals %in% train_levels)] <- "Unknown"
    test_df[[p]] <- factor(test_vals, levels = train_levels)
  }
}

#Select only relevant columns and drop any rows with NA
train_df <- train_df %>%
  select(
    id, Gender, Age,        # raw Age
    AgeGroup2,              # binned Age
    IgM,
    Area, AreaType, HouseType,  # separate features
    District, Area_House,   # plus interaction
    Outcome
  ) %>%
  na.omit()

test_df <- test_df %>%
  select(
    id, Gender, Age,
    AgeGroup2,
    IgM,
    Area, AreaType, HouseType,
    District, Area_House
  ) %>%
  na.omit()



#Step 4: Split into training and validation
set.seed(12345)
train_index <- createDataPartition(train_df$Outcome, p = 0.80, list = FALSE)
model_train <- train_df[train_index, ]
model_valid <- train_df[-train_index, ]


#Step 5: Identify and drop one‐level factors
factor_cols  <- names(model_train)[sapply(model_train, is.factor)]
factor_cols  <- setdiff(factor_cols, "Outcome")
level_counts <- sapply(model_train[factor_cols], nlevels)

cat("Levels per factor in model_train:\n")
print(level_counts)

one_level_cols <- names(level_counts[level_counts < 2])
cat("Factors with <2 observed levels (will be dropped):\n")
print(one_level_cols)

# Build reduced predictor list by removing any factor with only one level
all_predictors <- c(
  "Gender", "Age", "AgeGroup2", "IgM",
  "Area", "AreaType", "HouseType", "District", "Area_House"
)
reduced_predictors <- setdiff(all_predictors, one_level_cols)

cat("Using these predictors:\n")
print(reduced_predictors)


#Step 6: Build formula
formula_reduced <- as.formula(paste("Outcome ~", paste(reduced_predictors, collapse = " + ")))
cat("Formula to be used:\n")
print(formula_reduced)


#Step 7: Set up train control with upsampling
train_ctrl <- trainControl(
  method           = "cv",
  number           = 5,
  sampling         = "up",
  classProbs       = TRUE,
  summaryFunction  = twoClassSummary
)


#Step 8:Model training

# 8a) Logistic Regression

train_ctrl <- trainControl(method="cv", number=5)

set.seed(12345)
model_logistic <- train(
  Outcome ~ .,
  data = model_train,
  method = "glm",
  trControl = train_ctrl,
  metric = "Accuracy"
)
print(model_logistic)
cat("Logistic CV Accuracy (best):", max(model_logistic$results$Accuracy), "\n\n")

# 8b) Random Forest

p <- ncol(model_train) - 1
rf_grid <- expand.grid(mtry = unique(pmax(1, round(sqrt(p) + (-2:2)))))

set.seed(12345)
model_rf <- train(
  Outcome ~ .,
  data = model_train,
  method = "rf",
  tuneGrid = rf_grid,
  ntree = 500,
  trControl = train_ctrl,
  metric = "Accuracy"
)

print(model_rf)
cat("Random Forest CV Accuracy (best):", max(model_rf$results$Accuracy), "\n\n")

# Compare logistic vs RF CV accuracies
log_acc <- max(model_logistic$results$Accuracy)
rf_acc  <- max(model_rf$results$Accuracy)

best_model <- ifelse(rf_acc >= log_acc, "rf", "logistic")
cat("The best model is:", best_model, "\n\n")


# Step 9: Evaluate on validation set

cat("Validation set Outcome distribution:\n")
print(table(model_valid$Outcome))

# Recreate Area_House in validation set (same way as training)
model_valid <- model_valid %>%
  mutate(
    Area_House = interaction(AreaType, HouseType, drop = TRUE)
  )

# Align Area_House levels with training
model_valid$Area_House <- as.character(model_valid$Area_House)
model_valid$Area_House[!(model_valid$Area_House %in% levels(model_train$Area_House))] <- NA
model_train$Area_House <- forcats::fct_expand(model_train$Area_House, "Missing")
model_valid$Area_House <- factor(model_valid$Area_House, levels = levels(model_train$Area_House))
model_valid$Area_House <- forcats::fct_explicit_na(model_valid$Area_House, na_level = "Missing")

if (best_model == "rf") {
  val_probs <- predict(model_rf, newdata = model_valid, type = "prob")[, "Positive"]
} else {
  val_probs <- predict(model_logistic, newdata = model_valid, type = "prob")[, "Positive"]
}

cat("Validation set Positive probability summary:\n")
print(summary(val_probs))

hist(val_probs, breaks = 20, main = "val_probs distribution", xlab = "P(Positive)")


best_cut <- 0.5
best_acc <- 0
for (c in seq(0.10, 0.90, by = 0.01)) {
  preds <- factor(
    ifelse(val_probs >= c, "Positive", "Negative"),
    levels = c("Negative", "Positive")
  )
  cm <- confusionMatrix(preds, model_valid$Outcome, positive = "Positive")
  acc <- cm$overall["Accuracy"]
  if (acc > best_acc) {
    best_acc <- acc
    best_cut <- c
  }
}
cat("Optimal cutoff on validation:", best_cut, "with accuracy:", best_acc, "\n\n")


preds_best <- factor(ifelse(val_probs >= best_cut, "Positive", "Negative"),
                     levels = c("Negative","Positive"))
cat("Validation predictions sınıf dağılımı (best_cut):\n")
print(table(preds_best))

#Step 10: Retrain best model on all of train_df

# Repeat feature engineering on full_train
full_train <- train_df %>%
  mutate(
    Outcome    = factor(Outcome, levels = c("Negative", "Positive")),
    AgeGroup2  = cut(
      Age,
      breaks = c(-Inf, 12, 18, 35, 60, Inf),
      labels = c("Child", "Teen", "YoungAdult", "Adult", "Senior")
    ),
    Area_House = paste(AreaType, HouseType, sep = "_")
  )

# Fill NA in categoricals
for (p in cats) {
  if (p %in% names(full_train)) {
    full_train[[p]] <- as.character(full_train[[p]])
    full_train[[p]][is.na(full_train[[p]])] <- "Unknown"
    full_train[[p]] <- factor(full_train[[p]])
  }
}

# Impute numeric NAs (IgM)
if ("IgM" %in% reduced_predictors && any(is.na(full_train$IgM))) {
  median_igm <- median(full_train$IgM, na.rm = TRUE)
  full_train$IgM[is.na(full_train$IgM)] <- median_igm
}

#Drop leftover NA rows
full_train <- full_train %>%
  select(all_of(c(reduced_predictors, "Outcome"))) %>%
  na.omit()

# Retrain final model on full_train
set.seed(12345)
if (best_model == "rf") {
  # Retrain RF with best mtry
  final_model <- randomForest(
    formula    = formula_reduced,
    data       = full_train,
    mtry       = model_rf$bestTune$mtry,
    ntree      = 500,
    importance = TRUE
  )
  cat(
    "Final RF OOB Accuracy on full_train:",
    1 - final_model$err.rate[nrow(final_model$err.rate), "OOB"],
    "\n\n"
  )
} else {
  # Retrain logistic on full_train
  final_model <- glm(
    formula = formula_reduced,
    data    = full_train,
    family  = binomial(link = "logit")
  )
  roc_obj <- roc(full_train$Outcome, predict(final_model, type = "response"))
  cat("Final Logistic AUC on full_train:", auc(roc_obj), "\n\n")
}


# Step 11: Generate predictions on the test set

# Repeat feature engineering on test_df
full_test <- test_df %>%
  mutate(
    AgeGroup2  = cut(
      Age,
      breaks = c(-Inf, 12, 18, 35, 60, Inf),
      labels = c("Child", "Teen", "YoungAdult", "Adult", "Senior")
    ),
    Area_House = paste(AreaType, HouseType, sep = "_")
  )

# Align factor levels with full_train
for (p in cats) {
  if (p %in% names(full_train) && p %in% names(full_test)) {
    train_levels <- levels(full_train[[p]])
    test_vals    <- as.character(full_test[[p]])
    test_vals[!(test_vals %in% train_levels)] <- "Unknown"
    full_test[[p]] <- factor(test_vals, levels = train_levels)
  }
}

# Predict with chosen model and best_cutoff
if (best_model == "rf") {
  test_probs <- predict(final_model, newdata = full_test, type = "prob")[, "Positive"]
} else {
  test_probs <- predict(final_model, newdata = full_test, type = "response")
}

test_pred_class <- ifelse(test_probs >= best_cut, "Positive", "Negative")
test_pred_num   <- ifelse(test_pred_class == "Positive", 1, 0)

# Prepare submission
submission <- tibble(
  id      = full_test$id,
  Outcome = test_pred_num
)

stopifnot(nrow(submission) == nrow(full_test))

# Write CSV
write_csv(submission, "412..submission.csv")
cat("Submission file '412..submission.csv' has been written.\n\n")

head(submission)

