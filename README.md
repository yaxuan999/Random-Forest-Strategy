1. 每次main_func只能回测一种模型
2. 只用修改insample_train函数，添加method即可，请确保输出一致：y_test（不用改）,estimator_（模型）,estimator_train（训练集概率）,estimator_test（测试集概率）
3. method可用简写，写完后请在main_func中添加对应的method_name（全称）
4. 最后可用Auc_compare_all,Df_tier_ret_all将输出合并并保存csv（可以通过assign来记录参数改变）
5. 下载数据并解压到ipynb的同一目录下,https://drive.google.com/file/d/1s86HOE65NbThUssc99r-zHgskpkhGoNH/view?usp=sharing
