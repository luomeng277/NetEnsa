# 设置case工作目录
setwd("/home/LuoM/code/pythonProject4/course/newt2d/control_dir")


# 加载包
library(igraph)
library(psych)

# 使用list.files()函数读取文件夹中的文件名称

file_names <- list.files()

# 设置文件夹名称

output_folder <- "../net_control_dir_spearman"
dir.create(output_folder, showWarnings = FALSE)
# ...之前的代码...

# 循环处理每个文件
for (file_name in file_names) {
  # 读取数据
  df <- read.table( file_name, header = TRUE, row.names = 1, na.strings = "")
  df[is.na(df)] <- 0
  
  # 将数据转换为数据框
  df <- as.data.frame(df)
  # 禁用警告信息并计算OTU间两两相关系数矩阵
  suppressWarnings({
    occor <- corr.test(t(df), use = "pairwise", method = "spearman", adjust = "fdr", alpha = .05)
    occor.r <- occor$r # 取相关性矩阵R值
    occor.p <- occor$p # 取相关性矩阵p值
  })
  
  # 确定物种间存在相互作用关系的阈值，将相关性R矩阵内不符合的数据转换为0
  occor.r[occor.p > 0.05 | abs(occor.r) < 0.1] <- 0
  occor.r[is.na(occor.r)] <- 0
  
  # 设置对角矩阵的元素为1
  diag(occor.r) <- 1
  
  # 确保矩阵是对称的，将下三角矩阵的值复制到上三角矩阵
  occor.r <- occor.r + t(occor.r) - diag(occor.r)
  
  # 输出结果
  write.table(occor.r, paste(output_folder, "/", file_name, sep = ""), sep = "\t", row.names = TRUE, col.names = TRUE)
}

# ...之后的代码...
# # ...之后的代码...
# # 设置control工作目录
# setwd("D:/luom/NIFDN/control_dir") 
# 
# 
# # 使用list.files()函数读取文件夹中的文件名称
# file_names <- list.files("D:/luom/NIFDN/control_dir")
# 
# 
# # 设置文件夹名称
# output_folder <- "../net_control_dir_spearman"
# 
# dir.create(output_folder, showWarnings = FALSE)
# 
# # 循环处理每个文件
# for (file_name in file_names) {
#   # 读取数据
#   df <- read.table( file_name, header = TRUE, row.names = 1, na.strings = "")
#   df[is.na(df)] <- 0
#   
#   # 将数据转换为数据框
#   df <- as.data.frame(df)
#   # 禁用警告信息并计算OTU间两两相关系数矩阵
#   suppressWarnings({
#     occor <- corr.test(t(df), use = "pairwise", method = "spearman", adjust = "fdr", alpha = .05)
#     occor.r <- occor$r # 取相关性矩阵R值
#     occor.p <- occor$p # 取相关性矩阵p值
#   })
#   
#   # 确定物种间存在相互作用关系的阈值，将相关性R矩阵内不符合的数据转换为0
#   occor.r[occor.p > 0.05 | abs(occor.r) < 0.1] <- 0
#   occor.r[is.na(occor.r)] <- 0
#   
#   # 输出结果
#   write.table(occor.r, paste(output_folder, file_name, sep = "/"), sep = "\t", row.names = TRUE, col.names = TRUE)
# }
# 
