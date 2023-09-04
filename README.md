# wiseRL
WiseRL是一个面向分布式的强化学习框架，如下图所示，通过分布式的数据采样，集中训练的方式实现分布式的强化学习算法。
![image](https://user-images.githubusercontent.com/120070404/224656698-3ca6f4dc-53c6-452d-9035-560da132a8e1.png)<br/>
Env通过仿真进行环境采样<br/>
调用action的chose_action获得action<br/>
执行action，调用Learner的update更新参数<br/>
leaner通过fire调用action的load_mode,更新action参数<br/>

# 运行
<pre><code>
pip install -e .
cd example
python dqn.py
</code></pre>

