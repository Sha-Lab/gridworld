# Gridworld Environment for [SynPo]()

This repository implements a multi-task gridworld environment used in the SynPo paper. If you are using it to your research, please cite:

```
@inproceedings{hu2018synthesize,
  title={Synthesized Policies for Transfer and Adaptation across Tasks and Environments},
  author={Hu, Hexiang and Chen, Liyu and Gong, Boqing and Sha, Fei},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1176--1185},
  year={2018}
}
```

## Requirements

- Python 3+
- Numpy 1.10.0+
- OpenAI Gym
- Imageio
- Dill
- Matplotlib
- Pygame

## Usage
To use the GridWorld, you need to specify the the set of maps you want to sampled from and the number of objects appear in each map. The task is to pick up a sequence of objects in a specific order.

To get a gridworld env with mask observation, use GridWorld. If you want gridworld with RGB observation, use PORGBEnv wrapper. You can specify argument `l` to control the window size around the agent. If `l` is smaller than the radius of the map, the environment will be partially observable.

You can also call `env.get_opt_action()` to get the optimal action for the task currently sovling, which can be used easily for imitation learning.


## References

- [SynPo: Synthesized Policies
for Transfer and Adaptation across Environments and Tasks](https://sites.google.com/view/neurips2018-synpo/home)

## License
BISON is MIT licensed, as found in the LICENSE file.
