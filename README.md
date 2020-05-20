model1 ：sub1 用初始标签训练  50000数据  直接随机  没有调类别比例（0 ite）
model2 ：sub2 用初始标签训练   50000数据  直接随机  没有调类别比例（0 ite）
model3 ：sub3 用初始标签训练   50000数据  直接随机  没有调类别比例（0 ite）
model4-6  可以忽略
model 7-9 1 ite
model10-12 2 ite
model 13-15 3 ite
model16-18 4 ite
model 19-21 用奇怪的修改方式 correct  1ite
model 22-24 用奇怪的修改方式 correct  2ite
model 25-27 直接用merge 1ite


model 31-39  用水体指数加 mask   0-2ite
model 40-42 只用sar   （0ite）






train_sub1 sub2 sub3   150000训练数据随机划分为三个不重叠的子数据集  各50000

train_sub_01  水占0.1 共10000（03、05、。。。一个意思，表示水占总数居的比例）
