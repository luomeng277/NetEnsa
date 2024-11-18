# 设置case工作目录
setwd("/home/LuoM/code/pythonProject4/course/newt2d/case_dir")


# 使用list.files()函数读取文件夹中的文件名称
file_names <- list.files()

# 设置文件夹名称
folder_name <- "../net_case_dir_pearson"
# folder_name <- "../net_control_dir_pearson"

# 使用dir.create()函数创建文件夹
dir.create(folder_name)

# 将缺失值替换为0，计算Pearson相关系数及p值，并设置p值小于0.05的阈值
p_threshold <- 0.05

# 循环处理每个文件
for (file_name in file_names) {
  # 读取数据
  df <- read.table(file_name, header = TRUE, row.names = 1, na.strings = "")
  df[is.na(df)] <- 0
  
  # 将数据转换为数据框
  df <- as.data.frame(df)
  
  # 禁用警告信息并计算Pearson相关系数
  suppressWarnings({
    net_df <- round(cor(t(df), method = "pearson"), 8)
    net_df[is.na(net_df)] <- 0
    
    # 输出结果
    write.table(net_df, paste(folder_name, file_name, sep = "/"), sep = "\t", row.names = TRUE, col.names = TRUE)
  })
}
