# wiseRL
wiseRL是基于Ray Actor写的一套分布式强化学习算法。
设置三个角色，分别是：
Env
Action
Learner
![image](https://user-images.githubusercontent.com/120070404/224656698-3ca6f4dc-53c6-452d-9035-560da132a8e1.png)<br/>
Env通过仿真进行环境采样<br/>
调用action的chose_action获得action<br/>
执行action，调用Learner的update更新参数<br/>
leaner通过fire调用action的load_mode,更新action参数<br/>



