记录我对多任务学习问题的代码
目前学会的方法
1.直接将不同task的损失线性相加（这个就不要写代码了吧？）
2.gradnorm，计算训练过程中各个任务的收敛速度，重新定义权重减少不同任务的梯度范数差距。
3.MGDA，思路大概get了，代码很不好写，魔改了paper代码但是运行速度很慢。
