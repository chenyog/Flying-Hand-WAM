# 场景设置 
randomization的object的是可以绕z轴旋转调整的，只要能放在shelf上就可以，位置也随机，避免不同object重叠，同时保证涉及到task的shelf对应的空间不被占用就可以

当task的具体脚本发生变化后，同步修改task description的full description

当修改了_base_task后，检查所有其他task是否发生冲突

task、randomized objects 的z轴位置和xy位置都需要考虑到./description/objects_description里面的z_offset和radius等约束

# 约束条件
如果在任务执行期间，任意task object掉落在地面，任务算失败

# 轨迹设置
##分段轨迹
所有任务start -> pre -> grasp 是一段完整minco
所有任务grasp -> pull -> place_pre -> place 是一段完整minco
对于有多个抓取的任务，每一次从place -> place_pre -> pre -> grasp 是一段完整minco

## 时间分配
每一段MINCO轨迹和内部不同waypoint之间的时间分配根据任务进行调整，尽量保证整个任务执行的平滑又不拖沓