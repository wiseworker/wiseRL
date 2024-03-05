# wiseRL
WiseRL是一个面向分布式的强化学习框架，如下图所示，通过分布式的数据采样，集中训练的方式实现分布式的强化学习算法。
![image](https://github.com/wiseworker/wiseRL/blob/main/doc/runner.PNG)<br/>
1. Runner 是主程序入口，负责从环境采样，调用远程的Agent实现分布式采样，集中训练。
2. Agent是智能算法，通过调用平台的提供的智能算法，用户可以不用编写代码，方便进行智能算法进行训练，平台支持（DQN、PPO、DDPG、自博弈，多智能等算法）。
# 使用说明
1. 主函数：通过makeRunner创建多个Runner启动，Runner可以自动运行在Ray集群中
```
if __name__=='__main__':
    runners = makeRunner(GymRunner,num=2)
    results =[]
    for runner in runners:
        ref = runner.run.remote()
        results.append(ref)
    ray.wait(results)
```


# 运行
<pre><code>
pip install -e .
pip install pettingzoo pygame
cd example
ray start --head --port=6379
python dqn.py
</code></pre>

