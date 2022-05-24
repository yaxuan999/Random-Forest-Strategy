1. 每次main_func可以回测多种模型，平权投票；
2. main_func中methods以list的形式输入，k是cross-validation的次数（默认为5，但k=5的多模型回测会导致时长较长，可调低一点）；
3. 输出的结果会储存到results中，包括每种参数下的图和以csv形式保存的全阶段的tier_ret和auc_score
4. 下载数据并解压到py的同一目录下,https://drive.google.com/file/d/1s86HOE65NbThUssc99r-zHgskpkhGoNH/view?usp=sharing
5. 部分results见，https://drive.google.com/file/d/183KCSz0PYOSseAUo08-mcOwznRfX6WYf/view?usp=sharing
