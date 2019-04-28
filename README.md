# Kaggle比赛利器XGBboost
- 简介
	- 如果你的机器学习集成模型表现得不是那么尽如人意，那就使用XGBoost吧。XGBoost算法现在已经成为很多数据工程师重要工具，也是很多数据挖掘比赛前10队伍常用的选择。
	- XGBoost(eXtreme Gradient Boosting)是Gradient Boosting算法的一个优化的版本（不了解Boosting算法的建议先去了解一下，这是一种简单粗暴的算法思想），它是一种算法，同时它也是Python下的一个工具包。（`pip install xgboost` 依赖scipy和numpy)
	- 构建一个XGBoost模型比较容易，几行代码就可以了，但是XGBoost之所以如此强大正是因为其参数不少，调参，永远是机器学习的一个难点。
- 优势
	- 正则化
		- GBM（Gradient Boosting Machine）的实现没有像XGBoost这样的正则化步骤，因此很多时候过拟合处理比较麻烦，而XGBoost以“正则化提升(regularized boosting)”技术而闻名。
	- 并行处理
		- XGBoost支持并行处理，相比GBM有了速度上的巨大提升。（但是，众所周知，Boosting是顺序处理的，如何并行可以去Google了解一下）。
	- 兼容性强
		- 可以处理底层的numpy和scipy数据，特别对于部分大数据集可以处理稀疏矩阵直接进行训练。
	- 灵活性强
		- 允许用户定义自定义优化目标和评价标准，它对模型的使用开辟了一个全新的维度，用户的处理不会受到任何限制。
		- 可以自动处理缺失值，避免了太过繁琐的预处理，用户需要提供一个和其它样本不同的值，然后把它作为一个参数传进去，以此来作为缺失值的取值。并且XGBoost在不同节点遇到缺失值时采用不同的处理方法，而且会学习未来遇到缺失值时的处理方法。
	- 内置交叉验证
		- XGBoost允许在每一轮Boosting迭代中使用交叉验证。因此，可以方便地获得最优Boosting迭代次数。GBM的网格搜索有个最大弊端，只能在用户给出的范围内进行寻值。
	- 支持续训练
		- 可以在训练过的模型上继续训练，这与sklearn类似。
- 参数详解
	- 按照作者思路，参数分为三类，如下。（具体见官方文档[https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html)）
		- 通用参数
			- 宏观控制，类似超参数。
		- Booster参数
			- 控制每一步booster。
		- 目标参数
			- 控制训练目标的表现。
	- 通用参数
		- booster
			- 每次迭代用的模型，树模型和线性模型两种，基本上都是使用树模型。
			- 'gbtree' or 'dart' and 'gblinear'
		- verbosity
			- 训练可视化控制。
			- 0 (silent), 1 (warning), 2 (info), 3 (debug)
		- nthread
			- 控制使用多少线程，不设置则使用系统极限线程数。
		- disable_default_eval_metric
			- 是否禁用默认评估指标，默认为0，大于0表示禁用。
		- num_pbuffer， num_feature
			- 这两个参数用户不需要输入，模型自动输入。
	- Tree Booster参数（线性和dart参数类似，参照文档）
		- eta
			- 默认0.3
			- 类似学习速率。
			- 通过减少每一步的权重，可以提高模型的鲁棒性。
			- 典型值为0.01-0.2。
			- 范围[0, 1]
		- gamma
			- 默认0
			- 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
			- 这个参数的值越大，算法越保守。**属于需要调整的重点参数**。
			- 范围[0,∞]
		- max_depth
			- 默认6
			- 树的最大深度
			- 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。需要使用CV来进行调优。
			- 典型值为3-10
			- 范围[0,∞]
		- min_child_weight
			- 默认1
			- 决定最小叶子节点样本权重和。
			- 也是用于避免过拟合，设置较大的值，避免模型学习局部特殊样本。但是太大也会欠拟合。
			- 范围[0,∞]
		- max_delta_step
			- 默认0
			- 这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。
			- 但是，这个参数一般不需要设置。但是当各类别的样本十分不平衡的时候，需要设置。
			- 范围[0,∞]
		- subsample
			- 默认1
			- 这个参数控制对于每棵树，随机采样的比例。
			- 减小这个参数的值，算法会更加保守，避免过拟合。这个值设置得过小，它可能会导致欠拟合。
			- 典型值为0.5-1
			- 范围(0,1]
		- colsample_bytree, colsample_bylevel, colsample_bynode
			- 子采样的一系列参数
		- lambda
			- 默认1
			- 权重的L2正则化项,增加使得模型保守。
		- alpha
			- 默认1
			- 权重的L1正则化项。
			- 应用在很高维度的情况下，可以使得算法的速度更快。
		- scale_pos_weight
			- 默认1
			- 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
		- tree_method
			- 详见文档
	- 目标参数
		- objective
			- 默认reg:squarederror
			- 这个参数指定学习对象和学习任务。
			- 常用值
				- binary:logistic 二分类的逻辑回归，返回预测的概率。
				- binary:logitraw 输出二分类得分。
				- multi:softmax 使用softmax的多分类器，返回预测的类别。
					- 在这种情况下，你还需要多设一个参数：num_class(类别总数目)。
				- multi:softprob 返回的是每个数据记录属于各个类别的概率。
		- eval_metric
			- 默认值取决于objective参数的取值
			- 对于验证数据的度量方法。
			- 对于回归问题，默认值是rmse，对于分类问题，默认值是error。
		- seed
			- 随机数种子，默认时间戳，为了复现训练结果，一般设定种子为固定数值。
- 实战讲解-波士顿房价预测
	- 在这个案例中我会比对几个一般模型和xgboost模型效果对比，以及一般的xgboost模型**调参过程**。（我会尽量使用不同于sklearn风格的xgboost独有的一些方法）
	- 数据使用sklearn自带的波士顿房价数据集。
	- 使用几个模型看一下效果。
		- 代码
			- ```python
				# 构建几个回归模型
				lr = LinearRegression()
				ridge = Ridge()
				dtr = DecisionTreeRegressor()
				
				xgbr = xgb.XGBRegressor()
				# 不调参看一下效果
				lr.fit(x_train, y_train)
				pred = lr.predict(x_valid)
				print("MSE:{} in LR".format(mean_squared_error(y_valid, pred)))
				
				ridge.fit(x_train, y_train)
				pred = ridge.predict(x_valid)
				print("MSE:{} in Ridge".format(mean_squared_error(y_valid, pred)))
				
				dtr.fit(x_train, y_train)
				pred = dtr.predict(x_valid)
				print("MSE:{} in DTR".format(mean_squared_error(y_valid, pred)))
				
				xgbr.fit(x_train, y_train)
				pred = xgbr.predict(x_valid)
				print("MSE:{} in XGBR".format(mean_squared_error(y_valid, pred)))
				```
		- 运行效果
			- 这就很尴尬，本来预想初始表现差，调参超越其他模型的。
			- ![](https://img-blog.csdnimg.cn/2019042820264170.png)
			- 那就在原来数据基础上提高吧。
	- 调参（搜参很耗费运算资源）
		- 支持sklearn网格和交叉，甚至sklearn有封装的XGBoost，这里没有使用。
		- 第一步，按照经验设定初始参数。
		- 第二步，**最佳迭代次数**的调整。
			- 代码及结果（后面过程类似）
				- ![](https://img-blog.csdnimg.cn/20190428203756388.png)
			- 可以修改粒度继续调参，将最优参数更新到参数中。
		- 第三步，调整min_child_weight以及max_depth。
			- 同样找到最优结果，填入参数中。
		- 第四步，搜索gamma。
		- 第五步，subsample以及colsample_bytree。
		- 第六步，reg_alpha以及reg_lambda。
		- 最后，learning_rate。
			- **还记得一开始设置了一个较大的学习率吗，现在是时候减小学习率了。
	- 使用调参后的模型
		- ![](https://img-blog.csdnimg.cn/2019042820581478.png)
		- 可以看到，确实有了改善（虽然很小，但是顶级回归赛前百名相差都是0.001级别的）
- 补充说明
	- 全部的代码可以在我的Github找到（使用Jupyter环境），欢迎star或者fork
	- 数据挖掘（特别是机器学习算法赛）关键还是特征工程，所以前期的特征工程必须做好，后面的调参只是“锦上添花”罢了。
	- 数据，决定了机器学习的上限，而算法，只是逼近这个上限。