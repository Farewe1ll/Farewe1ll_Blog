---
title: '高级统计方法 R 语言总结'
published: 2025-11-29
description: '高级统计方法 R 语言总结'
image: ''
tags: ['Statistical_Learning', 'R_Language']
category: 'Courses'
draft: false
lang: ''
series: 'Stat-Learning'
---

# 高级统计方法 R 语言总结

## 引入包

```r
library(ISLR)
```

统计学习中其他常用包：

```r
library(MASS) # 包含多种统计方法和数据集
library(randomForest) # 随机森林模型
library(e1071) # 支持向量机和其他机器学习方法
library(glmnet) # Lasso和Ridge回归
library(class) # K近邻算法
library(boot) # 自助法和其他统计方法
library(leaps) # 特征选择方法
library(pls) # 偏最小二乘回归
library(splines) # 样条回归
library(gam) # 广义加性模型
library(tree) # 决策树
library(gbm) # 梯度提升机
```

| 包名 | 作用 (根据来源) |
| :--: | :--: |
| **MASS** | 包含 `Boston` 等数据集；用于拟合线性判别分析（LDA）和二次判别分析（QDA）模型，通过 `lda()` 和 `qda()` 函数实现。 |
| **randomForest** | 用于实现装袋法（Bagging）和随机森林（Random Forest）；`randomForest()` 函数既可用于随机森林，也可用于装袋法（当 `mtry=p` 时）。 |
| **glmnet** | 用于拟合岭回归（Ridge regression）和 Lasso 模型；也可用于多项式逻辑回归的计算。 |
| **boot** | 包含 `boot()` 函数，用于通过反复地从数据集中有放回地抽取观测（自助抽样）来执行自助法。 |
| **splines** | 用于拟合回归样条的函数；其中的 `bs()` 函数可以产生针对给定结点的所有样条基函数的矩阵。 |
| **gam** | 包含 `s()` 函数，用于广义可加模型（GAM）中拟合光滑样条。 |
| **tree** | 用于构造分类树和回归树；包含 `tree()`、`cv.tree()`（交叉验证树的复杂性）和 `prune.misclass()`（剪枝）等函数。 |
| **gbm** | 用于建立回归树的提升法（Boosting），通过 `gbm()` 函数实现，常用于梯度提升模型。 |
| **e1071** | 包含支持向量机（SVM）等机器学习方法，通过 `svm()` 函数实现。 |
| **class** | 用于实现K近邻算法（K-Nearest Neighbors），通过 `knn()` 函数实现。 |
| **leaps** | 用于特征选择，包含 `regsubsets()` 函数实现逐步回归等方法。 |
| **pls** | 用于偏最小二乘回归（Partial Least Squares），包含 `plsr()` 和 `mvr()` 函数。 |

## 基本命令

### 变量赋值

```r
x <- 1 # 使用 <- 赋值
x = 1 # 使用 = 赋值
```

### 定义函数

```r
my_function <- function(arg1, arg2) {
	# 函数体
	result <- arg1 + arg2
	return(result)
}
```

### 基本类型

```r
TRUE # 逻辑型 TRUE
FALSE # 逻辑型 FALSE
# 向量
x <- c(1, 2, 3) # 创建一个数值向量
y <- c("a", "b", "c") # 创建一个字符向量
# 矩阵
m <- matrix(1:9, nrow=3, ncol=3) # 创建一个3x3矩阵
# 数据框
df <- data.frame(name=c("Alice", "Bob"), age=c(25, 30)) # 创建一个数据框
# 因子
f <- factor(c("low", "medium", "high")) # 创建一个因子变量
# 数组
a <- array(1:12, dim=c(3, 4)) # 创建一个3x4的数组
# 列表
lst <- list(name="Alice", age=25, scores=c(90, 95, 85)) # 创建一个列表
```

区别在于：

- 向量：一维数组，所有元素类型相同。

- 数组：多维数组，所有元素类型相同。

- 因子：用于分类数据，存储为整数并有标签。

- 列表：可以包含不同类型的元素。

- 矩阵：二维数组，所有元素类型相同。基本等同于二维数组。

- 数据框：二维表格，列可以有不同类型。

数据类型判断和转换：

```r
# 判断数据类型
is.numeric(x) # 检查是否为数值型
is.integer(x) # 检查是否为整数型
is.double(x) # 检查是否为双精度型
is.logical(x) # 检查是否为逻辑型
is.vector(x) # 检查是否为向量型
is.matrix(m) # 检查是否为矩阵型
is.list(lst) # 检查是否为列表型
is.na(x) # 检查是否为缺失值
is.nan(x) # 检查是否为非数值
is.function(my_function) # 检查是否为函数
is.factor(f) # 检查是否为因子型
is.character(y) # 检查是否为字符型
is.data.frame(df) # 检查是否为数据框
# 数据类型转换
as.numeric(f) # 将因子转换为数值型
as.integer(x) # 将数值型转换为整数型
as.double(x) # 将数值型转换为双精度型
as.logical(c(1, 0, 1)) # 将数值型转换为逻辑型
as.vector(m) # 将矩阵转换为向量型
as.matrix(df) # 将数据框转换为矩阵型
as.list(x) # 将向量转换为列表型
as.factor(y) # 将字符型转换为因子型
as.Date("2023-01-01") # 将字符型转换为日期型
as.character(x) # 将数值型转换为字符型
as.data.frame(m) # 将矩阵转换为数据框

na.omit(x) # 删除缺失值
```

## 常用函数

```r
mean(x) # 计算均值
median(x) # 计算中位数
var(x) # 计算方差
sd(x) # 计算标准差
summary(x) # 总结统计信息
cor(x, y) # 计算相关系数
length(x) # 获取向量长度
range(x) # 获取向量范围
rep(x, times = 3) # 重复向量元素
seq(from = 1, to = 10, by = 1) # 生成序列
scale(x, center = FALSE, scale = TRUE) # 标准化数据 默认都是TRUE
sample(x, size, replace = FALSE) # 从向量中随机抽样
sample(1:100, 100, replace = TRUE) # 从1到100中有放回地抽取100个样本
sample(100, 100, replace = T) # 从1到100中有放回地抽取100个样本
points(x, y, col = "red", pch = 19) # 在现有图形上添加点
cbind(x, y) # 按列合并矩阵或数据框
rbind(x, y) # 按行合并矩阵或数据框
sort(x) # 对向量进行排序
order(x) # 获取排序索引
unique(x) # 获取唯一值
table(x) # 生成混淆矩阵
dim(m) # 获取矩阵或数据框的维度
ls() # 列出当前环境中的对象
rm(object_name) # 删除对象
rnorm(n, mean = 0, sd = 1) # 生成正态分布随机数，n为数量，mean为均值，sd为标准差
set.seed(123) # 设置随机数种子，确保结果可重复
runif(n, min = 0, max = 1) # 生成均匀分布随机数，n为数量，min为最小值，max为最大值
plot(x, y, xlab = "X轴标签", ylab = "Y轴标签", main = "散点图标题") # 绘制散点图
plot(x, type = "l", col = "red", main = "折线图标题", xlab = "X轴标签", ylab = "Y轴标签") # 绘制折线图
plot(x, type = "b", col = "green", main = "折线图标题", xlab = "X轴标签", ylab = "Y轴标签") # 绘制带点的折线图
hist(x, breaks = 10, col = "blue", main = "直方图标题", xlab = "X轴标签") # 绘制直方图
boxplot(x, main = "箱线图标题", ylab = "Y轴标签") # 绘制箱线图

pdf("plot.pdf") # 将图形输出到PDF文件
plot(x, y) # 绘制图形
dev.off() # 关闭图形设备，保存文件

seq(from = 1, to = 10, by = 2) # 生成序列，从1到10，步长为2
```

## 数据操作

```r
data <- read.csv("data.csv") # 读取CSV文件为数据框
head(data) # 查看数据前几行
tail(data) # 查看数据后几行
str(data) # 查看数据结构
subset(data, column_name > 10) # 条件筛选数据
names(data) # 获取数据框的列名
summary(data) # 查看数据摘要统计
attach(data) # 附加数据框，允许直接访问列名
data$column_name # 访问数据框的某一列
data[1:10, ] # 访问数据框的前10行
data[, c("col1", "col2")] # 访问数据框的指定列
data[which(data$column_name > 10), ] # 条件筛选
data$new_column <- data$col1 + data$col2 # 创建新列
data <- na.omit(data) # 删除缺失值
write.csv(data, "cleaned_data.csv") # 写入CSV文件
```

## 矩阵操作

```r
A <- matrix(1:9, nrow=3, ncol=3) # 创建矩阵
B <- matrix(9:1, nrow=3, ncol=3) # 创建另一个矩阵
C <- A + B # 矩阵加
D <- A %*% B # 矩阵乘
tA <- t(A) # 矩阵转置
invA <- solve(A) # 矩阵求逆
eigenA <- eigen(A) # 矩阵特征值和特征向
detA <- det(A) # 矩阵行列式
A[-(c(1,3)), ] # 删除矩阵的第1和第3行
A[, -(2)] # 删除矩阵的第2列
```

## 简单线性回归拟合

```r
model1 <- lm(y ~ . - x + 0, data = data) # 拟合所有预测变量但是不包括 x 和截距的线性回归模型
model1_ <- lm(y ~ x, data = data, subset = train_indices) # 使用子集数据拟合线性回归模型
summary(model1) # 查看模型摘要
coef(model1) # 获取模型系数
confint(model1) # 获取系数的置信区间
plot(data$x, data$y) # 绘制散点图
abline(model1, col = "red") # 添加回归线
predictions <- predict(model1, newdata = predict_data, interval = "confidence/prediction/none") # 进行预测，给出置信区间/预测区间/预测值

par(mfrow = c(2, 2)) # 设置图形布局为2x2
plot(model1) # 绘制诊断图

model2 <- lm(y ~ poly(x, 2), data = data) # 拟合二次多项式回归模型
model2_ <- lm(y ~ I(x^2), data = data) # 拟合二次多项式回归模型
model3 <- lm(y ~ log(x), data = data) # 拟合对数回归模型

model4 <- lm(y ~ x1 + x2:x3, data = data) # 拟合多元线性回归模型（会自动添加主效应）
model5 <- lm(y ~ (x1 + x2 + x3)^2, data = data) # 所有主效应 + 所有两两交互


data$group <- as.factor(data$group)
model6 <- lm(y ~ x + group, data = data) # 包含分类变量的线性回归模型

anova(model, model2) # 比较两个嵌套模型的方差分析
```

## 逻辑斯谛回归拟合

```r
log.fit <- glm(y ~ x1 + x2, data = data, family = binomial) # 拟合逻辑斯谛回归模型
summary(log.fit) # 查看模型摘要
coef(log.fit) # 获取模型系数
predicted_probs <- predict(log.fit, newdata = predict_data, type = "response/link") # 预测概率/对数几率值
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0) # 根据概率进行分类
```

## 线性判别分析

```r
library(MASS)
lda.fit <- lda(class ~ x1 + x2, data = data) # 拟合线性判别分析模型
summary(lda.fit) # 查看模型摘要
predictions <- predict(lda.fit, newdata = predict_data) # 进行预测
predicted_classes <- predictions$class # 获取预测类别
posterior_probs <- predictions$posterior # 获取后验概率
```

## 二次判别分析

```r
library(MASS)
qda.fit <- qda(class ~ x1 + x2, data = data) # 拟合二次判别分析模型
summary(qda.fit) # 查看模型摘要
predictions <- predict(qda.fit, newdata = predict_data) # 进行预测
predicted_classes <- predictions$class # 获取预测类别
posterior_probs <- predictions$posterior # 获取后验概率
```

## KNN 分类

```r
library(class)
predicted_classes <- knn(train = train_data, test = test_data, cl = train_labels, k = 3) # 使用KNN进行分类，k为邻居数
```

## K-fold 和 Bootstrap

```r
library(boot)
set.seed(123) # 设置随机数种子以确保结果可重复
cv_error <- cv.glm(data, model, K = 10)$delta[1] # 10折交叉验证误差
# cv.glm 还有一个 delta[2]，表示调整后的交叉验证误差
boot_fn <- function(data, indices) {
  fit <- lm(y ~ x1 + x2, data = data[indices, ])
  return(coef(fit))
}
set.seed(123) # 设置随机数种子以确保结果可重复
boot_results <- boot(data, boot_fn, R = 1000) # Bootstrap 估计，R为重复次数
boot_results$t0 # 原始样本的系数估计（与自助无关）
boot_results$t # Bootstrap 样本的系数估计（为一个矩阵，每行对应一个Bootstrap样本的系数估计）
boot.ci(boot_results, type = "bca", index = 2) # 获取第2个系数的置信区间
```

## 最优子集选择、向前逐步选择和向后逐步选择

```r
library(leaps)
best_subset <- regsubsets(y ~ x1 + x2 + x3, data = data, nvmax = 3) # 最优子集选择，nvmax为最大选择预测变量个数
summary(best_subset) # 查看选择结果摘要
plot(best_subset, scale = "adjr2") # 绘制调整后的 R^2 图
coef(best_subset, id = 2) # 获取包含2个预测变量的最佳模型的系数
# regsubsets 返回 which, rsq（最佳模型的 R^2）, rss, adjr2, bic, cp 等，不能直接用于 predict()

forward_step <- regsubsets(y ~ x1 + x2 + x3, data = data, nvmax = 3, method = "forward") # 向前逐步选择
backward_step <- regsubsets(y ~ x1 + x2 + x3, data = data, nvmax = 3, method = "backward") # 向后逐步选择

predict.regsubsets <- function(object, newdata, id) {
	form <- as.formula(object$call[[2]])
	mat <- model.matrix(form, newdata)
	coefi <- coef(object, id = id)
	xvars <- names(coefi)
	predictions <- mat[, xvars] %*% coefi
	return(predictions)
}
```

## 岭回归和Lasso

```r
library(glmnet)
x <- model.matrix(y ~ x1 + x2 + x3, data = data)[, -1] # 创建模型矩阵，去掉截距列
y <- data$y
ridge_model <- glmnet(x, y, alpha = 0, lambda = 0.1) # 拟合岭回归模型，alpha=0表示岭回归

lasso_model <- glmnet(x, y, alpha = 1, lambda = 0.1) # 拟合Lasso模型，alpha=1表示Lasso回归
cv_lasso <- cv.glmnet(x, y, alpha = 1) # 使用交叉验证选择Lasso的最佳lambda
best_lambda <- cv_lasso$lambda.min # 获取最佳lambda值
lasso_model_best <- glmnet(x, y, alpha = 1, lambda = best_lambda) # 使用最佳lambda拟合Lasso模型
```

## PCR 和 PLS 回归

```r
library(pls)
pcr_model <- pcr(y ~ x1 + x2 + x3, data = data, scale = TRUE, validation = "CV") # 拟合PCR模型，使用交叉验证
pls_model <- plsr(y ~ x1 + x2 + x3, data = data, scale = TRUE, validation = "CV") # 拟合PLS模型，使用交叉验证

validationplot(pcr_model, val.type = "MSEP", estimate = "CV") # 绘制PCR模型的均方误差图
```

## 非线性模型

### 多项式回归和阶梯函数

```r
model_poly <- lm(y ~ poly(x, 3), data = data) # 拟合三次多项式回归模型
model_step <- lm(y ~ cut(x, breaks = c(0, 10, 20, 30)), data = data) # 使用阶梯函数拟合模型
```

### 样条

```r
library(splines)
model_spline <- lm(y ~ bs(x, df = 4), data = data) # 使用B样条拟合模型，df为自由度
model_spline_degree2 <- lm(y ~ bs(x, degree = 2, df = 5), data = data) # 使用二次B样条拟合模型，degree为多项式的次数
model_spline_knots <- lm(y ~ bs(x, knots = c(10, 20)), data = data) # 使用指定结点的B样条拟合模型

model_ns <- lm(y ~ ns(x, df = 4), data = data) # 使用自然样条拟合模型
model_ns_knots <- lm(y ~ ns(x, knots = c(10, 20)), data = data) # 使用指定结点的自然样条拟合模型

model_smooth <- smooth.splines(x, y, df = 16) # 使用光滑样条拟合，等效自由度为16
model_smooth_cv <- smooth.splines(x, y, cv = TRUE) # 使用交叉验证选择光滑参数
```

### 局部回归

```r
model_loess <- loess(y ~ x, data = data, span = 0.5) # 使用局部回归拟合模型，span为平滑参数
```

### GAM

```r
library(gam)
gam_model <- gam(y ~ s(x1) + s(x2), data = data) # 拟合广义可加模型，s()表示光滑函数
```

## 决策树

### 分类树

```r
library(tree)
tree_model <- tree(class ~ x1 + x2, data = data) # 拟合分类树模型
summary(tree_model) # 查看模型摘要
plot(tree_model) # 绘制树结构
text(tree_model, pretty = 0) # 添加节点标签，pretty=0表示显示所有类别标签
predictions <- predict(tree_model, newdata = predict_data, type = "class") # 进行预测
cv.tree_model <- cv.tree(tree_model, FUN = prune.misclass) # 交叉验证选择剪枝参数，最小化分类错误率
pruned_tree <- prune.misclass(tree_model, best = 5) # 剪枝，best为最佳叶节点数，只剩下5个叶节点

cv.tree_model_dev <- cv.tree(tree_model, FUN = prune.tree) # 交叉验证选择剪枝参数，最小化偏差
pruned_tree_dev <- prune.tree(tree_model, best = 5) # 剪枝，只剩下5个叶节点
```

### 回归树

```r
library(tree)
reg_tree_model <- tree(y ~ x1 + x2, data = data) # 拟合回归树模型
summary(reg_tree_model) # 查看模型摘要
plot(reg_tree_model) # 绘制树结构
text(reg_tree_model, pretty = 0) # 添加节点标签
predictions <- predict(reg_tree_model, newdata = predict_data) # 进行预测
cv.reg_tree_model <- cv.tree(reg_tree_model) # 交叉验证选择剪枝参数，最小化偏差
pruned_reg_tree <- prune.tree(reg_tree_model, best = 5) # 剪枝，只剩下5个叶节点
```

### 装袋法和随机森林

```r
library(randomForest)
set.seed(123) # 设置随机数种子以确保结果可重复
bagging_model <- randomForest(y ~ x1 + x2 + x3, data = data, mtry = 3, importance = TRUE) # 装袋法，mtry=p
set.seed(123) # 设置随机数种子以确保结果可重复
rf_model <- randomForest(y ~ x1 + x2 + x3, data = data, mtry = 2, importance = TRUE) # 随机森林，mtry<p
importance(rf_model) # 查看变量重要性
varImpPlot(rf_model) # 绘制变量重要性图
```

### 提升法

```r
library(gbm)
set.seed(123) # 设置随机数种子以确保结果可重复
boosting_model <- gbm(y ~ x1 + x2 + x3, data = data, distribution = "gaussian", n.trees = 1000, interaction.depth = 4, shrinkage = 0.01, verbose = FALSE) # 提升法，拟合回归树，n.trees为树的数量，interaction.depth为树的最大深度，shrinkage为学习率
summary(boosting_model) # 查看变量重要性
predictions <- predict(boosting_model, newdata = predict_data, n.trees = 1000) # 进行预测

boosting_model_bin <- gbm(class ~ x1 + x2 + x3, data = data, distribution = "bernoulli", n.trees = 1000, interaction.depth = 4, shrinkage = 0.01, verbose = FALSE) # 提升法，拟合分类树
```

## SVM

### 支持向量分类器

```r
library(e1071)
svm_model <- svm(class ~ x1 + x2, data = data, kernel = "linear", cost = 1, scale = TRUE) # 拟合线性支持向量分类器，进行
summary(svm_model) # 查看模型摘要
predictions <- predict(svm_model, newdata = predict_data) # 进行预测

tune_result <- tune(svm, class ~ x1 + x2, data = data, kernel = "linear", ranges = list(cost = c(0.1, 1, 10))) # 调参，选择最佳cost值
best_model <- tune_result$best.model # 获取最佳模型
```

### 支持向量机

```r
library(e1071)
svm_model_rbf <- svm(class ~ x1 + x2, data = data, kernel = "radial", cost = 1, gamma = 0.5, scale = TRUE) # 拟合径向基函数支持向量机
summary(svm_model_rbf) # 查看模型摘要
predictions <- predict(svm_model_rbf, newdata = predict_data) # 进行预测
tune_result_rbf <- tune(svm, class ~ x1 + x2, data = data, kernel = "radial", ranges = list(cost = c(0.1, 1, 10), gamma = c(0.1, 0.5, 1))) # 调参，选择最佳cost和gamma值
best_model_rbf <- tune_result_rbf$best.model # 获取最佳模型
```